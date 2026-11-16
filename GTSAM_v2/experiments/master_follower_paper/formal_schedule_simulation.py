"""Common stochastic world and causal execution of frozen schedules."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import math
import time
from typing import Any, Callable, Sequence
import gtsam
import numpy as np

from estimator.communication import (
    FollowerInitialStateMessage, MasterNavigationPublisher, MasterPoseMessage,
    SourceOdomIncrement, SOURCE_ODOMETRY_PROVENANCE, MASTER_NAVIGATION_PROVENANCE,
)
from estimator.distributed_graphs import pose_key
from estimator.odometry_preintegration import compose_pose_covariance, preintegrate_continuous_trajectory
from .formal_candidate_schedules import ACTION_JOINT, FixedCandidateSchedule
from .formal_protocol import (
    FormalWorldDataset, PhysicalWave, JointProtocolCycle, SharedJointMeasurement,
    SharedTimestampMasterMeasurement, _build_event_timelines, _master_event_data,
    _sample_tangent_gaussian,
)
from .formal_schedule_compiler import compile_nominal_schedule
from .formal_sector_sweep_motion import build_structured_sector_sweep_case, sector_sweep_parameter_rows
from .formal_runner import metric_row
from .graphs import CYCLE_SHARED_TIMESTAMP, MasterFollowerDualGraphEstimator
from .model import OdomSegment

PROTOCOL_FIXED_SCHEDULE = "fixed_schedule"


@dataclass(frozen=True)
class FixedScheduleScenario:
    name: str
    config: Any
    schedule: FixedCandidateSchedule
    radial_rotation_rad: float = math.pi / 12.0
    joint_protocol: str = PROTOCOL_FIXED_SCHEDULE

    @property
    def baseline_cycle_count(self):
        return self.schedule.ordinary_action_count

    @property
    def joint_cycle_count(self):
        return self.schedule.joint_action_count


@dataclass
class ScheduleCase:
    seed: int
    datasets: dict
    compiled: dict
    schedules: dict
    audit_rows: list
    path_parameters: list
    shared_data_audit: dict
    generation_wall_s: float


def _signature(wave):
    return wave.sender_id, int(round(wave.send_time * 10**10))


def generate_schedule_case(seed: int, schedules: Sequence[FixedCandidateSchedule], *,
                           progress_callback: Callable | None = None) -> ScheduleCase:
    """Sample shared atoms once, independent of schedule evaluation order."""
    started = time.perf_counter()
    notify = progress_callback or (lambda phase: None)
    candidate_ids = tuple(schedule.candidate_id for schedule in schedules)
    plans = {s.candidate_id: s for s in schedules}
    if not schedules or len(plans) != len(schedules):
        raise ValueError("one or more uniquely named schedules are required")
    for s in plans.values():
        s.validate()
        expected = tuple(math.pi/12 + i*math.pi/3 for i in range(6))
        if any(abs(math.atan2(math.sin(a-b), math.cos(a-b))) > 1e-10
               for a, b in zip(s.sector_angles_rad, expected)):
            raise ValueError("frozen geometry must be 15 degrees")
    notify("trajectory")
    spec, profiles, _, noise_seed = build_structured_sector_sweep_case(seed=seed, orientation_deg=15.)
    cfg = spec.config

    @lru_cache(maxsize=50000)
    def pose_at(agent, t):
        return profiles[agent].pose_at(float(t))

    notify("compile_physical_waves")
    compiled = {
        name: compile_nominal_schedule(plans[name], cfg, pose_at)
        for name in candidate_ids
    }
    canonical = {}
    for item in compiled.values():
        for wave in item.waves:
            old = canonical.setdefault(_signature(wave), wave)
            if any(abs(old.receive_time(a)-t)>1e-9 for a,t in wave.receive_times):
                raise AssertionError("equivalent physical wave arrival mismatch")
    signatures = sorted(canonical, key=lambda x: (x[1], x[0]))
    global_id = {s:i for i,s in enumerate(signatures)}
    raw = {global_id[s]: replace(canonical[s], wave_id=global_id[s], protocol=PROTOCOL_FIXED_SCHEDULE)
           for s in signatures}
    timelines, grid, tokens = _build_event_timelines(cfg, raw)
    master_refs, master_messages = _master_event_data(seed, noise_seed, cfg, raw, pose_at)
    physical = {}
    for wid, w in raw.items():
        sender = master_refs[wid] if w.sender_id in cfg.master_ids else tokens[(w.sender_id,"SEND",wid)]
        physical[wid] = PhysicalWave(wid, PROTOCOL_FIXED_SCHEDULE, w.sender_id, w.send_time,
            w.relay_id, w.receive_times, sender,
            tuple((a,tokens[(a,"RECEIVE",wid)]) for a,_ in w.receive_times if a in cfg.follower_ids))
    prior_rng = np.random.RandomState(noise_seed*1009+11)
    priors = {}
    for agent in cfg.follower_ids:
        e = np.asarray([prior_rng.normal(0,cfg.follower_initial_xy_sigma_m),
                        prior_rng.normal(0,cfg.follower_initial_xy_sigma_m),
                        prior_rng.normal(0,cfg.follower_initial_heading_sigma_rad)])
        priors[agent] = FollowerInitialStateMessage(follower_id=agent,
            pose=pose_at(agent,0.).retract(e), covariance=cfg.follower_initial_covariance,
            timestamp=0.,source_epoch=seed)
    odom = {}
    digest = hashlib.sha256()
    for offset, agent in enumerate(cfg.follower_ids):
        notify(f"atomic_odometry:{agent}")
        rng = np.random.RandomState(noise_seed*1009+101+37*offset)
        segments = []
        for first,last in zip(timelines[agent],timelines[agent][1:]):
            mean,q = preintegrate_continuous_trajectory(first.timestamp,last.timestamp,cfg.continuous_noise,
                pose_at=lambda t,a=agent:pose_at(a,float(t)))
            delta = mean.retract(_sample_tangent_gaussian(q,rng))
            segment = OdomSegment(segment_id=f"{agent}:raw:{first.event_id}:{last.event_id}",
                agent_id=agent,start_event_id=first.event_id,end_event_id=last.event_id,
                start_time=first.timestamp,end_time=last.timestamp,delta_pose=delta,covariance=q)
            segments.append(segment)
            digest.update(segment.segment_id.encode())
            digest.update(np.asarray([first.timestamp,last.timestamp,delta.x(),delta.y(),delta.theta()],dtype='<f8').tobytes())
            digest.update(np.asarray(q,dtype='<f8').tobytes())
        odom[agent] = tuple(segments)
    notify("timestamp_measurements")
    rng = np.random.RandomState(noise_seed*1009+29)
    errors, tkeys = {}, {}
    for agent in cfg.follower_ids:
        for wid,wave in physical.items():
            if wave.sender_id != agent:
                errors[(agent,wid)] = float(rng.normal(0,cfg.timestamp_sigma_s))
                tkeys[(agent,wid)] = int(gtsam.symbol("T",len(tkeys)))
    truth = {a:np.asarray([[p.x(),p.y(),p.theta()] for p in
        (pose_at(a,i*cfg.sensor_dt_s) for i in range(cfg.step_count+1))]) for a in cfg.agent_ids}
    datasets, audits = {}, []
    for name in candidate_ids:
        local = {w.wave_id:physical[global_id[_signature(w)]] for w in compiled[name].waves}
        waves = {w.wave_id:w for w in local.values()}
        cycles, master_rows = [], []
        mid = 0
        for action in compiled[name].actions:
            intent = action.intent
            ws = tuple(local[w.wave_id] for w in action.waves)
            joint_rows, relay_row = [], None
            for listener in cfg.follower_ids:
                is_joint = intent.action_type == ACTION_JOINT and listener != intent.relay_id
                used = ws if is_joint else ws[:2]
                refs = tuple(w.receive_event(listener) for w in used)
                arrivals = np.asarray([ref.timestamp+errors[(listener,w.wave_id)] for ref,w in zip(refs,used)])
                keys = tuple(tkeys[(listener,w.wave_id)] for w in used)
                if is_joint:
                    m = SharedJointMeasurement(mid,intent.sequence_index,intent.anchor_master_id,
                        intent.second_master_id,intent.relay_id,listener,ws,refs,arrivals,
                        np.asarray([0.,0.,action.anchor_wait_s]),np.eye(4)*cfg.timestamp_sigma_s**2,
                        keys,cfg.sound_speed_mps)
                    joint_rows.append(m)
                else:
                    m = SharedTimestampMasterMeasurement(mid,intent.sequence_index,intent.block_id,
                        intent.anchor_master_id,intent.second_master_id,listener,
                        intent.action_type==ACTION_JOINT,used,refs,arrivals,0.,keys,cfg.sound_speed_mps)
                    master_rows.append(m)
                    if intent.action_type==ACTION_JOINT:
                        relay_row=m
                mid+=1
            if intent.action_type==ACTION_JOINT:
                cycles.append(JointProtocolCycle(intent.sequence_index,intent.relay_id,ws,tuple(joint_rows),
                    action.anchor_wait_s,intent.block_id,intent.anchor_master_id,intent.second_master_id,relay_row))
        pairs = {(a,wid) for a,wid in errors if wid in waves}
        dataset = FormalWorldDataset(seed,noise_seed,FixedScheduleScenario(spec.name,cfg,plans[name]),cfg,
            profiles,truth,timelines,grid,odom,priors,waves,(),tuple(cycles),tuple(master_rows),
            {wid:master_messages[wid] for wid in waves if wid in master_messages},
            {p:tkeys[p] for p in sorted(pairs)}, {p:errors[p] for p in sorted(pairs)})
        datasets[name]=dataset
        for agent in cfg.follower_ids:
            nj=sum(m.listener_id==agent for m in dataset.joint_measurements)
            nm=sum(m.listener_id==agent for m in master_rows)
            nr=sum(c.relay_id==agent for c in cycles)
            if nj+nm != plans[name].ordinary_action_count+plans[name].joint_action_count:
                raise AssertionError("direct row budget mismatch")
            if nr != dict(plans[name].relay_quotas)[agent]:
                raise AssertionError("relay quota mismatch")
            usages=[k for m in (*dataset.joint_measurements,*master_rows) if m.listener_id==agent for k in m.timestamp_error_keys]
            coalesced_pose_slots=sum(
                len(m.logical_event_refs)-len({int(ref.key) for ref in m.logical_event_refs})
                for m in dataset.joint_measurements
                if m.listener_id==agent
            )
            audits.append(dict(seed=seed,method=name,agent_id=agent,orientation_deg=15.,
                physical_wave_count=len(waves),joint_cycle_count=len(cycles),ordinary_action_count=plans[name].ordinary_action_count,
                relay_count=nr,direct_master_rows=nj+nm,peer_rows=2*nj,
                unique_timestamp_prior_count=len(set(usages)),timestamp_prior_reuse_count=len(usages)-len(set(usages)),
                coalesced_joint_pose_slot_count=coalesced_pose_slots,
                expected_master_prior_count=len(dataset.master_messages),schedule_fingerprint=plans[name].fingerprint))
    notify("route_statistics")
    parameters=sector_sweep_parameter_rows(datasets[candidate_ids[0]])
    audit=dict(atomic_odom_sha256=digest.hexdigest(),atomic_odom_segment_count=sum(map(len,odom.values())),
               unique_potential_wave_count=len(physical),candidate_count=len(candidate_ids),shared_atomic_odometry=True,
               shared_initial_state_messages=True,shared_physical_timestamp_realizations=True)
    pose_at.cache_clear()
    return ScheduleCase(seed,datasets,compiled,plans,audits,parameters,audit,time.perf_counter()-started)


def _aggregate_segments(dataset):
    required={a:{ref.event_id for i,ref in enumerate(dataset.grid_event_refs[a]) if i%10==0}
              for a in dataset.config.follower_ids}
    for m in (*dataset.joint_measurements,*dataset.joint_master_measurements):
        required[m.listener_id].update(ref.event_id for ref in m.listener_events)
    for c in dataset.joint_cycles:
        required[c.relay_id].add(c.waves[2].sender_event.event_id)
    result={}
    for a in dataset.config.follower_ids:
        atoms={s.start_event_id:s for s in dataset.odom_segments[a]}
        chosen=sorted(required[a]); segments=[]
        for first,last in zip(chosen,chosen[1:]):
            cursor=first; delta=gtsam.Pose2(); q=np.zeros((3,3))
            while cursor<last:
                segment=atoms[cursor]
                q=compose_pose_covariance(q,segment.delta_pose,segment.covariance)
                delta=delta.compose(segment.delta_pose); cursor=segment.end_event_id
            if cursor!=last:
                raise AssertionError("noncontiguous atomic aggregation")
            segments.append(OdomSegment(segment_id=f"{a}:aggregate:{first}:{last}",agent_id=a,
                start_event_id=first,end_event_id=last,start_time=atoms[first].start_time,
                end_time=segment.end_time,delta_pose=delta,covariance=q))
        result[a]=tuple(segments)
    return result


def run_schedule_method(case: ScheduleCase,candidate_id: str):
    started=time.perf_counter(); dataset=case.datasets[candidate_id]; plan=case.schedules[candidate_id]
    active=_aggregate_segments(dataset)
    estimators={a:MasterFollowerDualGraphEstimator(dataset,a,include_remote_initial_states=bool(plan.joint_action_count))
                for a in dataset.config.follower_ids}
    events=[]
    def add(t,priority,kind,payload):
        events.append((t,priority,len(events),kind,payload))
    for segments in active.values():
        for s in segments: add(s.end_time,0,'segment',s)
    for w in dataset.waves.values():
        add(w.send_time,1,'send',w)
        for receiver,t in w.receive_times:
            if receiver in estimators: add(t,2,'receive',(w,receiver))
    for m in (*dataset.joint_measurements,*dataset.joint_master_measurements): add(m.completion_time,3,'measurement',m)
    for t in range(1,201): add(float(t),4,'evaluate',t*10)
    packets={}; transmitted=set(); early=0; rows=[]
    previous={a:0. for a in estimators}; source_seen={a:set() for a in estimators}
    for t,_,_,kind,payload in sorted(events):
        try:
            if kind=='segment': estimators[payload.agent_id].add_local_segment(payload)
            elif kind=='send':
                w=payload
                if w.wave_id in transmitted: raise AssertionError('duplicate physical wave')
                transmitted.add(w.wave_id)
                if w.sender_id in dataset.config.master_ids:
                    p=MasterNavigationPublisher.publish(dataset.master_messages[w.wave_id])
                else:
                    p=estimators[w.sender_id].make_source_increment(w.sender_event.event_id)
                    if p is None or p.end_time>t+1e-10: raise AssertionError('noncausal source increment')
                    if abs(p.start_time-previous[w.sender_id])>1e-10: raise AssertionError('source gap')
                    if source_seen[w.sender_id].intersection(p.segment_ids): raise AssertionError('source overlap')
                    source_seen[w.sender_id].update(p.segment_ids); previous[w.sender_id]=p.end_time
                packets[w.wave_id]=p
            elif kind=='receive':
                w,receiver=payload; p=packets[w.wave_id]; graph=estimators[receiver].cooperative
                if isinstance(p,MasterPoseMessage): graph.ingest_master_pose(p)
                else:
                    if pose_key(p.source_id,p.end_event_id) in graph.graph.known_keys: early+=1
                    graph.ingest_follower_increment(p)
            elif kind=='measurement':
                graph=estimators[payload.listener_id].cooperative
                if isinstance(payload,SharedJointMeasurement): graph.add_cycle(payload,CYCLE_SHARED_TIMESTAMP)
                else: graph.add_shared_timestamp_master_measurement(payload)
            elif kind=='evaluate':
                for a,e in estimators.items():
                    ref=dataset.grid_event_refs[a][payload]
                    pose,cov=e.cooperative.owner_state(ref.event_id)
                    row=metric_row(dataset,candidate_id,t,a,pose,cov)
                    truth=dataset.truth_pose(a,t)
                    row.update(truth_x_m=truth.x(),truth_y_m=truth.y(),truth_heading_rad=truth.theta(),orientation_deg=15.)
                    rows.append(row)
        except Exception as exc:
            raise RuntimeError(f'seed={case.seed} candidate={candidate_id} event={kind} t={t:.10f}: {exc}') from exc
    if len(transmitted)!=200 or early: raise AssertionError('wave budget or early shadow failure')
    ga=[]
    for a,e in estimators.items():
        g=e.cooperative
        expected=next(r for r in case.audit_rows if r['method']==candidate_id and r['agent_id']==a)
        actual=dict(direct_master_rows=g.joint_cycle_factor_count+g.master_only_factor_count,
                    peer_rows=2*g.joint_cycle_factor_count,unique_timestamp_prior_count=g.timestamp_prior_count,
                    expected_master_prior_count=g.master_prior_count)
        for key,v in actual.items():
            if v!=expected[key]: raise AssertionError(f'{candidate_id}/{a}/{key}: {v} != {expected[key]}')
        ga.append(dict(agent_id=a,**actual))
    sc=sum(isinstance(p,SourceOdomIncrement) and p.provenance!=SOURCE_ODOMETRY_PROVENANCE for p in packets.values())
    mc=sum(isinstance(p,MasterPoseMessage) and p.provenance!=MASTER_NAVIGATION_PROVENANCE for p in packets.values())
    if sc or mc: raise AssertionError('message provenance contamination')
    return rows,dict(seed=case.seed,method=candidate_id,orientation_deg=15.,wall_time_s=time.perf_counter()-started,
        metric_rows=len(rows),physical_wave_count=len(transmitted),unique_physical_wave_count=len(transmitted),
        ordinary_action_count=plan.ordinary_action_count,joint_cycle_count=plan.joint_action_count,
        direct_master_row_count=sum(r['direct_master_rows'] for r in ga),peer_row_count=sum(r['peer_rows'] for r in ga),
        master_prior_count=sum(r['expected_master_prior_count'] for r in ga),timestamp_prior_count=sum(r['unique_timestamp_prior_count'] for r in ga),
        source_contamination_count=int(sc),master_contamination_count=int(mc),shadow_created_before_arrival=early,
        source_gap_count=0,source_overlap_count=0,duplicate_timestamp_prior_count=0,duplicate_master_prior_count=0,
        raw_odom_segment_count=case.shared_data_audit['atomic_odom_segment_count'],
        active_odom_segment_count=sum(map(len,active.values())),schedule_fingerprint=plan.fingerprint,agent_audits=ga)

# MADNESS Runtime Architecture

## Overview
- `world.h` owns the runtime hub: `World` embeds `WorldMpiInterface& mpi`, `WorldAmInterface& am`, `WorldTaskQueue& taskq`, `WorldGopInterface& gop` (src/madness/world/world.h:203-206). It maps global object IDs via `uniqueidT` and `ConcurrentHashMap` (world.h:276-305) and resolves worlds by ID (world.h:264-288).
- MPI layer: `WorldMpiInterface` wraps `SafeMPI::Intracomm` and initialization/finalization (`worldmpi.h:268-454`), constructed inside `World`.
- Transport: `RMI` provides the low-level server/send layer (`worldrmi.h:171-399`), used by active messages.
- Active messages: `AmArg` and `WorldAmInterface` serialize handlers and payloads, tagged with `worldid` for dispatch (`worldam.h:63-162, 212`); included by `world.h`, `worldref.h`, and used by GOP.
- Tasking and futures: `WorldTaskQueue` enqueues work and returns `Future` results (`world_task_queue.h:325, 524-534`); `Future`/`FutureImpl` manage async results and depend on remote references (`future.h:108-735`).
- Distributed references: `WorldPtr` holds globally addressable pointers (`worldptr.h:73-398`); `RemoteReference` manages remote refcounts using AM and `ConcurrentHashMap` caches (`worldref.h:197, 399-600`).
- Collectives: `WorldGopInterface` implements reductions/broadcasts/barriers on top of task queue + AM (`worldgop.h:152-1547`), exposed via `World::gop`.
- Supporting IDs/types/profiling: `ProcessID`/`Tag` (worldtypes.h:48-49), `uniqueidT` (uniqueid.h:57-116), profiling stats for `print_stats(World&)` (worldprofile.h:30-80; world.h:89-105).

## Diagram
Regenerate during docs build to avoid drift.
```mermaid
graph TD
  World[World (world.h)] --> MPI[WorldMpiInterface (worldmpi.h)]
  World --> AM[WorldAmInterface (worldam.h)]
  World --> TQ[WorldTaskQueue (world_task_queue.h)]
  World --> GOP[WorldGopInterface (worldgop.h)]
  AM --> RMI[RMI (worldrmi.h)]
  TQ --> Future[Future (future.h)]
  Future --> RemoteRef[RemoteReference / WorldPtr (worldref.h / worldptr.h)]
  GOP --> AM
  GOP --> TQ
  AM --> World
```

## How Data Moves
- Submit task: client uses `World::taskq` APIs; `WorldTaskQueue::reduce` shows task submission returning `Future<resultT>` (world_task_queue.h:524-534).
- Result handling: `Future`/`FutureImpl` track completion and may carry remote references (`future.h:108-735`), which use AM to manage refcounts (`worldref.h:399-600`).
- Sending work: AM serializes handlers and payloads into `AmArg` (worldam.h:108-142) with `worldid`; RMI transports it (`worldrmi.h:171-399`); target resolves `World` via `World::world_from_id` (world.h:264-288).
- Execution: handler deserializes via `AmArg::make_input_arch` (worldam.h:135-148), executes, and fulfills futures back over AM/RMI.
- Collectives: `WorldGopInterface` uses task queue + AM for reductions/broadcasts (worldgop.h:152-1547), leveraging per-world communicator in `WorldMpiInterface` (`worldmpi.h:268-454`).
- IDs and object lookup: `uniqueidT` generated/registered in `World::register_ptr` (world.h:276-305); handlers can resolve local objects with `World::ptr_from_id` (world.h:318-332) when receiving AMs.

## Index Table
- `world.h` (`World`): referenced by `worldam.h`, `world_task_queue.h`, `future.h`, `worldptr.h`.
- `worldmpi.h` (`WorldMpiInterface`): stored in `World` (world.h:203).
- `worldrmi.h` (`RMI`): used by `worldam.h` for transport.
- `worldam.h` (`WorldAmInterface`, `AmArg`): referenced by `World` (world.h:204), `worldref.h` (include for AM allocation), `worldgop.h` (active messages).
- `world_task_queue.h` (`WorldTaskQueue`): referenced by `World` (world.h:205), `world_object.h`, `worldgop.h`.
- `future.h` (`Future`, `FutureImpl`): returned from task queue (world_task_queue.h:524-534); depends on `worldref.h`, `world.h`.
- `worldref.h` (`RemoteReference`): used by `future.h`, `world_object.h`; depends on `worldam.h`, `worldhashmap.h`.
- `worldptr.h` (`WorldPtr`): used by `worldref.h` and archives; includes `world.h` for IDs.
- `worldgop.h` (`WorldGopInterface`): exposed via `World::gop` (world.h:206); uses task queue and AM.
- `worldprofile.h` (`ProfileStat`): used by `print_stats(World&)` (world.h:89-105).

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_TASK_H
#define HK_TASK_H

#include <Common/Base/Object/hkReferencedObject.h>


/// Value used to indicate that there is no appropriate elf to process a given task.
#if !defined(HK_PLATFORM_SIM)
#	define HK_INVALID_ELF	HK_NULL
#else 
// On the simulator we use the task type instead of the elf pointer.
#	define HK_INVALID_ELF	((void*)(0xFFFFFFFF))
#endif


/// A task is a self-contained unit of work that may be executed in any thread (worker or main).
/// Dependencies between tasks can be defined using a task graph (hkTaskGraph).
class hkTask : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE_CLASS );
	
		/// Process the task. When this method returns the task is considered as finished and the dependencies of 
		/// other tasks on it fulfilled.
		virtual void process() = 0;

		/// Return the address of the SPU elf that processes this task, or HK_INVALID_ELF if processing on SPU is
		/// not supported (default).
		/// In the SPU simulator the task type (see hkTaskType) must be returned instead of the elf pointer.
		virtual void* getElf() { return HK_INVALID_ELF; }
};


/// This task type enum is used in the SPU simulator to dispatch tasks to the appropriate elf (see threadMain() in 
/// hkSpuSimulatorMain.cpp). It may be used as well to create unique frame IDs for the check determinism util (see 
/// hkCheckDeterminismUtil::Fuid). If your task does not require any of these things then it does not need to have
/// a task type here.
struct hkTaskType
{
	enum
	{
		// We start assigning values from 100 on to avoid overlapping with the values defined in hkJobType
		JOB_TYPE_MAX = 99,

		// Physics tasks
		HKNP_NARROW_PHASE_TASK,
		HKNP_SOLVER_TASK,
		HKNP_COLLISION_QUERY_TASK,
		HKNP_RAYCAST_TASK,

		// Destruction tasks
		HKND_CONTROLLER_TASK,
		HKND_PROJECT_MANIFOLD_TASK,
	};
};


#endif // HK_TASK_H

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */

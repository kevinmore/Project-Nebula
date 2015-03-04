/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_STEP_INPUT_H
#define HKNP_STEP_INPUT_H

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Pool/hkThreadPool.h>

class hkBlockStreamAllocator;


/// Input to hknpWorld::stepXxx() methods.
struct hknpStepInput
{
	//+hk.MemoryTracker(ignore=True)

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpStepInput );

		/// Constructor. Sets default values.
		HK_FORCE_INLINE hknpStepInput();

		/// Helper method to calculate the number of CPU/SPU threads available,
		/// based on a thread pool and the current platform.
		HK_FORCE_INLINE void calcNumberOfThreads( const hkThreadPool* threadPool );

	public:

		/// The time step by which to advance the simulation. Defaults to 1/30th of a second.
		hkReal m_deltaTime;

		/// Block stream allocator for step local allocations, such as Jacobians and events.
		/// The memory should be 128 byte aligned.
		/// Typically you should allow 1K per active object, however for complex objects the memory requirement
		/// can rise higher than this.
		/// Note that on PS3 only the fixed block stream allocator is supported.
		hkBlockStreamAllocator* m_stepLocalStreamAllocator;

		/// Number of CPU threads available to process simulation tasks, including the main thread.
		/// Defaults to 0. Must be set to at least 1.
		int m_numCpuThreads;

		/// Number of SPU threads available to process simulation tasks on PlayStation(R)3.
		/// Defaults to 0.
		int m_numSpuThreads;
};

#include <Physics/Physics/Dynamics/World/hknpStepInput.inl>


#endif // HKNP_STEP_INPUT_H

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

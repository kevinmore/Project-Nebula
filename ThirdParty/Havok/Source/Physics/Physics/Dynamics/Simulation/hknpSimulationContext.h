/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SIMULATION_CONTEXT_H
#define HKNP_SIMULATION_CONTEXT_H

#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>

class hkThreadPool;
class hknpEventDispatcher;
struct hkTaskGraph;


/// Internal helper struct to provide context information to the simulation step.
class hknpSimulationContext
{
	public:

#if defined(HK_PLATFORM_XBOX360)
		enum { MAX_NUM_THREADS = 6 };
#elif defined(HK_PLATFORM_PS3)
		enum { MAX_NUM_THREADS = 2+6 };	// 2 PPU, 6 SPU
#elif defined(HK_PLATFORM_PS4)
		enum { MAX_NUM_THREADS = 8 };
#else
		enum { MAX_NUM_THREADS = 12 };
#endif

	public:

		/// Get the current thread's index.
		static int HK_CALL getCurrentThreadNumber();

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSimulationContext );

		/// Set up the context, before a simulation step.
		void init( hknpWorld* world, const hknpStepInput& stepInput, hkBlockStreamAllocator* tempAlloator );

		/// Clear the context, after a simulation step.
		void clear();

		/// Get the number of CPU/PPU threads.
		HK_FORCE_INLINE int getNumCpuThreads() const;

		/// Get the number of SPU threads.
		HK_FORCE_INLINE int getNumSpuThreads() const;

		/// Get the total number of threads (CPU/PPU + SPU).
		HK_FORCE_INLINE int getNumThreads() const;

		/// Get the current thread's context.
		hknpSimulationThreadContext* getThreadContext();

		/// Get the given thread's context.
		HK_FORCE_INLINE hknpSimulationThreadContext* getThreadContext( int threadIndex );

		/// Dispatch all commands in the command queue, including deferred commands.
		void dispatchCommands( hkPrimaryCommandDispatcher* dispatcher = HK_NULL );

		/// Dispatch only those commands which have the HKNP_POST_COLLIDE_COMMAND_BIT bit set.
		/// Copy any other commands to the deferred command queue.
		void dispatchPostCollideCommands( hkPrimaryCommandDispatcher* dispatcher, hknpEventDispatcher* eventDispatcher );

	#if defined(HK_PLATFORM_HAS_SPU)
		/// Get a list of PPU virtual table addresses for each shape type.
		HK_FORCE_INLINE void** getShapeVTablesPpu();
	#endif

	public:

		/// The task graph to which simulation tasks are added.
		hkTaskGraph* m_taskGraph;

	private:

		/// Number of CPU/PPU threads.
		int m_numCpuThreads;

		/// Number of SPU threads.
		int m_numSpuThreads;

		/// Thread specific contexts, one per thread.
		/// Contexts for SPU threads, when present, appear after the CPU ones.
		hknpSimulationThreadContext m_threadContexts[MAX_NUM_THREADS];

		/// A grid for deterministic command dispatch.
		hknpCommandGrid m_commandGrid;

	#if defined(HK_PLATFORM_HAS_SPU)
		/// Grid for deterministic dispatch of commands originated from SPU tasks.
		/// Commands here are executed after the ones in m_commandGrid.
		hknpCommandGrid m_spuCommandGrid;
	#endif

		/// A container for commands deferred until post solve.
		hkBlockStream<hkCommand> m_deferredCommandStream;
		hkThreadLocalBlockStreamAllocator* m_deferredCommandStreamAllocator;

	#if defined(HK_PLATFORM_HAS_SPU)
		/// A list of PPU virtual table addresses for each shape type.
		void** m_shapeVTablesPpu;
	#endif
};

#include <Physics/Physics/Dynamics/Simulation/hknpSimulationContext.inl>

#endif // HKNP_SIMULATION_CONTEXT_H

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

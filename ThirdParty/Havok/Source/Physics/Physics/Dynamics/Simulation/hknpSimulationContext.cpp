/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationContext.h>

#include <Physics/Physics/Dynamics/Simulation/Utils/hknpSimulationDeterminismUtil.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>
#include <Physics/Physics/Collide/Shape/VirtualTableUtil/hknpShapeVirtualTableUtil.h>
#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>


extern HK_THREAD_LOCAL( int ) hkThreadNumber;

/*static*/ int HK_CALL hknpSimulationContext::getCurrentThreadNumber()
{
	int threadId = HK_THREAD_LOCAL_GET( hkThreadNumber );
	return threadId;
}


void hknpSimulationContext::init(
	hknpWorld* world, const hknpStepInput& stepInfo, hkBlockStreamAllocator* tempAllocator )
{
	const int numThreads = stepInfo.m_numCpuThreads + stepInfo.m_numSpuThreads;

	HK_ASSERT2( 0xad838340, stepInfo.m_numCpuThreads > 0, "Number of CPU threads must be at least 1." );
	HK_ASSERT3( 0xad838341, 0 <= numThreads && numThreads <= MAX_NUM_THREADS,
		"Number of threads must be at most " << MAX_NUM_THREADS << "." );

	m_numCpuThreads = stepInfo.m_numCpuThreads;
	m_numSpuThreads = stepInfo.m_numSpuThreads;
	m_taskGraph = HK_NULL;	// set from outside later

	// Set up command grid
#if (HK_CONFIG_THREAD == HK_CONFIG_SINGLE_THREADED)
	m_commandGrid.setSize( 1 );
#else
	m_commandGrid.setSize( world->m_spaceSplitter->getNumLinks() );
	HK_ON_PLATFORM_HAS_SPU( m_spuCommandGrid.setSize( world->m_spaceSplitter->getNumLinks() ) );
#endif

	// Set up thread contexts
	for( int i = 0; i < numThreads; i++ )
	{
		const bool isCpuThread = i < stepInfo.m_numCpuThreads;
		hknpCommandGrid* commandGrid = &m_commandGrid;
#if defined(HK_PLATFORM_HAS_SPU)
		if (!isCpuThread)
		{
			commandGrid = &m_spuCommandGrid;
		}
#endif
		m_threadContexts[i].init( world, commandGrid, tempAllocator, i, isCpuThread );
	}

	// Don't set up the deferred command stream and allocator until we need it
	m_deferredCommandStreamAllocator = HK_NULL;

#if defined(HK_PLATFORM_PPU)
	// Get PPU shape v-table pointers
	m_shapeVTablesPpu = hknpShapeVirtualTableUtil::getVTables();
#endif
}

void hknpSimulationContext::clear()
{
	int threadNumber = getCurrentThreadNumber();
	hknpSimulationThreadContext& threadContext = m_threadContexts[threadNumber];

	// Shutdown other threads
	for( int i = 0; i < getNumThreads(); i++ )
	{
		if( i != threadNumber )
		{
			hknpSimulationThreadContext& tl = m_threadContexts[i];
			tl.shutdownThreadContext( threadContext.m_tempAllocator, threadContext.m_heapAllocator );
		}
	}

	// Free deferred command stream
	if( m_deferredCommandStreamAllocator )
	{
		m_deferredCommandStream.clear( m_deferredCommandStreamAllocator );
		delete m_deferredCommandStreamAllocator;
		m_deferredCommandStreamAllocator = HK_NULL;
	}

	// Shutdown this thread
	threadContext.shutdownThreadContext( threadContext.m_tempAllocator, threadContext.m_heapAllocator );

	m_numCpuThreads = 0;
	m_numSpuThreads = 0;
	m_taskGraph = HK_NULL;
}

hknpSimulationThreadContext* hknpSimulationContext::getThreadContext()
{
	int threadId = HK_THREAD_LOCAL_GET( hkThreadNumber );
	HK_ASSERT( 0xf0fdad10, threadId >= 0 && threadId <= m_numCpuThreads );
	return &m_threadContexts[threadId];
}

void hknpSimulationContext::dispatchCommands( hkPrimaryCommandDispatcher* dispatcher )
{
	if( dispatcher == HK_NULL )
	{
		dispatcher = m_threadContexts[0].m_world->m_commandDispatcher;
	}

	// Finalize all command writers
	for( int i = 0; i < m_numCpuThreads; i++ )
	{
		hknpSimulationThreadContext& tl = m_threadContexts[i];
		tl.finalizeCommandWriters();
	}

	// Dispatch deferred commands, if any
	if( m_deferredCommandStreamAllocator )
	{
		hkBlockStream<hkCommand>::Reader reader;
		reader.setToStartOfStream( &m_deferredCommandStream );
		for( const hkCommand* com = reader.access(); com; com = reader.advanceAndAccessNext( com ) )
		{
			hknpSimulationDeterminismUtil::check(com);
			dispatcher->exec( *com );
		}
	}

	// Dispatch per-thread commands
	m_commandGrid.dispatchCommands( dispatcher );
#if defined(HK_PLATFORM_HAS_SPU)
	m_spuCommandGrid.dispatchCommands( dispatcher );
#endif
}

void hknpSimulationContext::dispatchPostCollideCommands(
	hkPrimaryCommandDispatcher* dispatcher, hknpEventDispatcher* eventDispatcher )
{
	hknpSimulationThreadContext* threadContext = getThreadContext();

	// Finalize all command writers.
	for( int i = 0; i < m_numCpuThreads; i++ )
	{
		m_threadContexts[i].finalizeCommandWriters();
	}

	// Set up deferred command stream
	if( !m_deferredCommandStreamAllocator )
	{
		
		
		hkBlockStreamAllocator* tempAllocator = threadContext->m_tempAllocator->m_blockStreamAllocator;
		m_deferredCommandStreamAllocator = new hkThreadLocalBlockStreamAllocator( tempAllocator, -1 );
		m_deferredCommandStream.initBlockStream( m_deferredCommandStreamAllocator );
	}

	// Create a writer to move all deferred commands to m_deferredCommandStream.
	hkBlockStream<hkCommand>& deferredCommands = m_deferredCommandStream;
	hkBlockStream<hkCommand>::Writer deferredCommandWriter;
	deferredCommandWriter.setToEndOfStream( m_deferredCommandStreamAllocator, &deferredCommands );

	if( eventDispatcher )
	{
		eventDispatcher->m_commandWriter = &deferredCommandWriter;
	}

	// Process commands from the command grid.
	hkBlockStream<hkCommand>::Reader reader;

#if defined(HK_PLATFORM_HAS_SPU)
	hknpCommandGrid* commandGrids[] = { &m_commandGrid, &m_spuCommandGrid };
	for (int gridIndex = 0; gridIndex < 2; ++gridIndex)
	{
		hknpCommandGrid& commandGrid = *commandGrids[gridIndex];
#else
	{
		hknpCommandGrid& commandGrid = m_commandGrid;
#endif
		for (int gi = 0; gi < commandGrid.m_entries.getSize(); ++gi)
		{
			const hkBlockStreamBase::LinkedRange* range = &(commandGrid[gi]);

			if (!range->isEmpty())
			{
				do
				{
					reader.setToRange(range);

					for (const hkCommand* com = reader.access(); com; com = reader.advanceAndAccessNext( com ))
					{
						hknpSimulationDeterminismUtil::check(com);

						// Dispatch non-deferred command immediately.
						if ( com->m_filterBits & hknpCommandDispatchType::AS_SOON_AS_POSSIBLE )
						{
							dispatcher->exec( *com );
						}
						// Move deferred command.
						else
						{
							deferredCommandWriter.write16( com );
						}
					}

					range = range->m_next;
				} while (range);
				commandGrid[gi].clearRange();
			}
		}
	}

	deferredCommandWriter.finalize();

	if( eventDispatcher )
	{
		eventDispatcher->m_commandWriter = HK_NULL;
	}

	// Reset all thread command streams.
	// We cannot do this using while reading using a consumer, because the streams might contain linked ranges.
	for( int i = 0; i < getNumThreads(); ++i )
	{
		hknpSimulationThreadContext& otherTl = m_threadContexts[i];
		otherTl.resetCommandStreamAndWriter( threadContext->m_tempAllocator, threadContext->m_heapAllocator );
	}
}

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

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>

// this
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>
#include <Physics/Physics/Dynamics/Simulation/Utils/hknpSimulationDeterminismUtil.h>

hknpSimulationThreadContext::hknpSimulationThreadContext()
{
	HK_ON_DEBUG(m_heapAllocator = HK_NULL;)
	HK_ON_DEBUG(m_currentGridEntryDebug = -1;)
}


void hknpSimulationThreadContext::init(
	hknpWorld* world, hknpCommandGrid* commandGrid, hkBlockStreamAllocator* tempAllocator,
	int threadIdx, bool setupCommandStreamWriter )
{
	hknpSimulationThreadContext& tl = *this;
	HK_ASSERT( 0xf034dedf, m_heapAllocator == HK_NULL );

	tl.m_world = world;
	tl.m_modifierManager = world->getModifierManager();
	tl.m_shapeTagCodec = world->getShapeTagCodec();

	// Get qualities
	{
		tl.m_qualities = world->getBodyQualityLibrary()->getBuffer();
		tl.m_numQualities = world->getBodyQualityLibrary()->getCapacity();
	}

	// Get materials
	{
		tl.m_materials = world->getMaterialLibrary()->getBuffer();
		tl.m_numMaterials = world->getMaterialLibrary()->getCapacity();
	}

	tl.m_deactivationData = HK_NULL;
	tl.m_spaceSplitterData = HK_NULL;
	tl.m_heapAllocator = new hkThreadLocalBlockStreamAllocator( world->m_persistentStreamAllocator, -1 );

	tl.m_tempAllocator = HK_NULL;
	if( tempAllocator )
	{
		if( tempAllocator == world->m_persistentStreamAllocator )
		{
			tl.m_tempAllocator = tl.m_heapAllocator;
		}
		else
		{
			tl.m_tempAllocator = new hkThreadLocalBlockStreamAllocator( tempAllocator, threadIdx );
		}
	}
	m_commandBlockStream.initBlockStream( tl.m_heapAllocator, true );	// needed for SPU

	m_commandGrid = commandGrid;
	m_commandWriter = HK_NULL;
	if( setupCommandStreamWriter )
	{
		m_commandWriter = new hkBlockStreamCommandWriter;
		m_commandWriter->m_writer.setToStartOfStream( tl.m_heapAllocator, &m_commandBlockStream );
	}

	m_triangleShapePrototypes[0] = m_triangleShapePrototype0.getTriangleShape();
	m_triangleShapePrototypes[1] = m_triangleShapePrototype1.getTriangleShape();

	HK_ON_CPU( tl.m_heapAllocator->m_threadId = threadIdx );
}


void hknpSimulationThreadContext::resetCommandStreamAndWriter( hkThreadLocalBlockStreamAllocator* tempAllocator, hkThreadLocalBlockStreamAllocator* heapAllocator )
{
	m_commandBlockStream.reset( heapAllocator );

	// Reset the command writer if there is one (i.e. in non-SPU threads)
	if (m_commandWriter)
	{
		m_commandWriter->m_writer.setToStartOfStream( this->m_heapAllocator, &m_commandBlockStream );
		m_currentGridEntryRange.setStartPoint(&m_commandWriter->m_writer);
	}
}

void hknpSimulationThreadContext::shutdownThreadContext(hkThreadLocalBlockStreamAllocator* tempAllocator, hkThreadLocalBlockStreamAllocator* heapAllocator )
{
	hknpSimulationThreadContext& tl = *this;

	delete m_commandWriter;
	m_commandWriter = HK_NULL;
	m_commandBlockStream.clear( heapAllocator );	// note: if you get an assert here, you might have forgotten to call appendCommands or dispatch Commands

	// if temp allocator is different than the m_heap, delete it
	if ( tl.m_tempAllocator != tl.m_heapAllocator )
	{
		delete tl.m_tempAllocator;
	}
	delete tl.m_heapAllocator;

	tl.m_tempAllocator = HK_NULL;
	tl.m_heapAllocator = HK_NULL;
}

void hknpSimulationThreadContext::appendCommands( hkBlockStream<hkCommand>* externalCommandStream )
{
	if ( m_commandWriter )
	{
		m_commandWriter->m_writer.finalize();
	}

	if ( !m_commandBlockStream.isEmpty() )
	{
		externalCommandStream->append( m_tempAllocator, &m_commandBlockStream );
	}
}

void hknpSimulationThreadContext::dispatchCommands( hkPrimaryCommandDispatcher* dispatcher )
{
	if (m_commandGrid)
	{
		if ( m_commandWriter )
		{
			m_commandWriter->m_writer.finalize();
		}
		if ( !dispatcher )
		{
			dispatcher = m_world->m_commandDispatcher;
		}

		m_commandGrid->dispatchCommands( dispatcher );
	}
}


void hknpCommandGrid::dispatchCommands( hkPrimaryCommandDispatcher* dispatcher )
{
	// Iterate over the grid and execute it.
	hkBlockStream<hkCommand>::Reader reader;

	for (int gi = 0; gi < m_entries.getSize(); ++gi)
	{
		hkBlockStreamBase::LinkedRange* range = &(m_entries[gi]);

		if (!range->isEmpty())
		{
			do
			{
				reader.setToRange(range);

				for (const hkCommand* com = reader.access(); com; com = reader.advanceAndAccessNext( com ))
				{
					hknpSimulationDeterminismUtil::check(com);
					dispatcher->exec( *com );
				}
				range = range->m_next;

			} while (range);
		}
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

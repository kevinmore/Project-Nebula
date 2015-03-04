/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/BroadPhase/hknpBroadPhaseViewer.h>

#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/Shape/hkDisplayAABB.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>


int hknpBroadPhaseViewer::s_tag = 0;

void HK_CALL hknpBroadPhaseViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpBroadPhaseViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hknpBroadPhaseViewer( contexts );
}


hknpBroadPhaseViewer::hknpBroadPhaseViewer( const hkArray<hkProcessContext*>& contexts )
:	hknpViewer( contexts ),
	m_showPreviousAabbs( false ),
	m_bufferSize( 32 * 1024 )
{

}

void hknpBroadPhaseViewer::setPreviousAabbsEnabled( bool enabled )
{
	m_showPreviousAabbs = enabled;
}

void hknpBroadPhaseViewer::setBufferSize( int size )
{
	HK_ASSERT2(0x16bc78ba, size >= (int)sizeof(hkDisplayAABB), "Buffer size too small" );
	m_bufferSize = size;
}

void hknpBroadPhaseViewer::step( hkReal deltaTime )
{
	if( !m_context || (m_context->getNumWorlds() == 0) )
	{
		return;
	}

	HK_TIMER_BEGIN( "BroadPhaseViewer", this );

	// Allocate a buffer of display objects.
	// This avoids large varying size allocations when there are lots of bodies in a world.
	const int bufferCapacity = m_bufferSize / sizeof(hkDisplayAABB);
	hkLocalBuffer<hkDisplayAABB> displayAabbs( bufferCapacity );
	hkLocalBuffer<hkDisplayGeometry*> displayAabbPtrs( bufferCapacity );
	{
		for( int i=0; i<bufferCapacity; ++i )
		{
			displayAabbPtrs[i] = new (&displayAabbs[i]) hkDisplayAABB();
		}
	}

	for( int wi=0; wi<m_context->getNumWorlds(); wi++ )
	{
		hknpWorld* world = m_context->getWorld(wi);

		// Borders
		{
			const hkAabb& aabb = world->m_intSpaceUtil.m_aabb;
			displayAabbs[0].setExtents( aabb.m_min, aabb.m_max );

			hkArray<hkDisplayGeometry*> displayGeometries( displayAabbPtrs.begin(), 1, bufferCapacity );
			m_displayHandler->displayGeometry( displayGeometries, hkColor::rgbFromChars( 240, 200, 0, 200 ), 0, s_tag );
		}

		// Previous AABBs
		if( m_showPreviousAabbs )
		{
			const hkArray<hkAabb16>& prevAabbs = world->m_bodyManager.getPreviousAabbs();

			int bufferIndex = 0;
			for( hknpBodyIterator it = world->getBodyIterator(); it.isValid(); it.next() )
			{
				const hknpBody& body = it.getBody();
				if( body.isAddedToWorld() )
				{
					hkAabb aabb;
					world->m_intSpaceUtil.restoreAabb( prevAabbs[ it.getBodyId().value() ], aabb );
					aabb.expandBy( -hkSimdReal_Inv_255 );	// shrink a little to avoid fighting with body AABB
					displayAabbs[bufferIndex].setExtents( aabb.m_min, aabb.m_max );
					if( ++bufferIndex == bufferCapacity )
					{
						// Flush
						hkArray<hkDisplayGeometry*> displayGeometries( displayAabbPtrs.begin(), bufferIndex, bufferCapacity );
						m_displayHandler->displayGeometry( displayGeometries, hkColor::YELLOW, 0, s_tag );
						bufferIndex = 0;
					}
				}
			}

			if( bufferIndex > 0 )
			{
				// Flush
				hkArray<hkDisplayGeometry*> displayGeometries( displayAabbPtrs.begin(), bufferIndex, bufferCapacity );
				m_displayHandler->displayGeometry( displayGeometries, hkColor::YELLOW, 0, s_tag );
			}
		}

		// Body AABBs
		{
			int bufferIndex = 0;
			for( hknpBodyIterator it = world->getBodyIterator(); it.isValid(); it.next() )
			{
				const hknpBody& body = it.getBody();
				if( body.isAddedToWorld() )
				{
					hkAabb aabb;
					world->m_intSpaceUtil.restoreAabb( body.m_aabb, aabb );
					displayAabbs[bufferIndex].setExtents( aabb.m_min, aabb.m_max );
					if( ++bufferIndex == bufferCapacity )
					{
						// Flush
						hkArray<hkDisplayGeometry*> displayGeometries( displayAabbPtrs.begin(), bufferIndex, bufferCapacity );
						m_displayHandler->displayGeometry( displayGeometries, hkColor::RED, 0, s_tag );
						bufferIndex = 0;
					}
				}
			}

			if( bufferIndex > 0 )
			{
				// Flush
				hkArray<hkDisplayGeometry*> displayGeometries( displayAabbPtrs.begin(), bufferIndex, bufferCapacity );
				m_displayHandler->displayGeometry( displayGeometries, hkColor::RED, 0, s_tag );
			}
		}
	}

	HK_TIMER_END();
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

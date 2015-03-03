/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/hknpProcessContext.h>

#include <Common/Visualize/hkProcessFactory.h>

#include <Physics/Physics/Extensions/Viewers/Constraint/hknpConstraintViewer.h>
#include <Physics/Physics/Extensions/Viewers/Manifold/hknpManifoldViewer.h>
#include <Physics/Physics/Extensions/Viewers/WeldingTriangle/hknpWeldingTriangleViewer.h>
#include <Physics/Physics/Extensions/Viewers/Shape/hknpShapeViewer.h>
#include <Physics/Physics/Extensions/Viewers/SubStep/hknpSubStepViewer.h>
#include <Physics/Physics/Extensions/Viewers/MassProperties/hknpMassPropertiesViewer.h>
#include <Physics/Physics/Extensions/Viewers/BroadPhase/hknpBroadPhaseViewer.h>
#include <Physics/Physics/Extensions/Viewers/Cell/hknpCellViewer.h>
#include <Physics/Physics/Extensions/Viewers/Deactivation/hknpDeactivationViewer.h>
#include <Physics/Physics/Extensions/Viewers/BodyId/hknpBodyIdViewer.h>
#include <Physics/Physics/Extensions/Viewers/BoundingRadius/hknpBoundingRadiusViewer.h>
#include <Physics/Physics/Extensions/Viewers/MotionTrail/hknpMotionTrailViewer.h>
#include <Physics/Physics/Extensions/Viewers/CompositeQueryAabb/hknpCompositeQueryAabbViewer.h>
#include <Physics/Physics/Extensions/Viewers/WorldSnapshot/hknpWorldSnapshotViewer.h>
#include <Physics/Physics/Extensions/Viewers/MotionId/hknpMotionIdViewer.h>


void HK_CALL hknpProcessContext::registerAllProcesses()
{
	hkProcessFactory& factory = hkProcessFactory::getInstance();

	// This is the order that the viewers appear in the VDB client, so put the more important/useful ones earlier
	hknpBodyIdViewer::registerViewer(factory);
	hknpMotionIdViewer::registerViewer(factory);
	hknpBroadPhaseViewer::registerViewer(factory);
	hknpShapeViewer::registerViewer(factory);
	hknpMassPropertiesViewer::registerViewer(factory);
	hknpConstraintViewer::registerViewer(factory);
	hknpManifoldViewer::registerViewer(factory);
	hknpMotionTrailViewer::registerViewer(factory);
	hknpBoundingRadiusViewer::registerViewer(factory);
	hknpDeactivationViewer::registerViewer(factory);
	hknpCellViewer::registerViewer(factory);
	hknpWeldingTriangleViewer::registerViewer(factory);
	hknpWorldSnapshotViewer::registerViewer(factory);
}


hknpProcessContext::hknpProcessContext()
{
	m_colorScheme = &m_defaultColorScheme;
}

hknpProcessContext::~hknpProcessContext()
{
	for (int w=(m_worlds.getSize()-1); w >=0 ; --w)
	{
		removeWorld( m_worlds[w] );
	}
}

void hknpProcessContext::setOwner( hkVisualDebugger* vdb )
{
	if (m_owner)
	{
		for (int wi=0;wi < m_worlds.getSize(); ++wi)
		{
			removeFromInspection(m_worlds[wi]);
		}
	}

	m_owner = vdb;

	if (vdb)
	{
		for (int i=0;i < m_worlds.getSize(); ++i)
		{
			addForInspection( m_worlds[i] );
		}
	}
}

void hknpProcessContext::addWorld( hknpWorld* newWorld )
{
	// make sure we don't have it already
	if ( m_worlds.indexOf( newWorld ) < 0 )
	{
		m_worlds.pushBack( newWorld );

		for ( int i=0; i < m_addListeners.getSize(); ++i )
		{
			m_addListeners[i]->worldAddedCallback( newWorld );
		}

		addForInspection( newWorld );

		HK_SUBSCRIBE_TO_SIGNAL( newWorld->m_signals.m_worldDestroyed, this, hknpProcessContext );
	}
}

void hknpProcessContext::removeWorld( hknpWorld* oldWorld )
{
	int wi = m_worlds.indexOf( oldWorld );
	if (wi >= 0)
	{
		oldWorld->m_signals.m_worldDestroyed.unsubscribeAll( this );

		for ( int i=0; i < m_addListeners.getSize(); ++i )
		{
			m_addListeners[i]->worldRemovedCallback( oldWorld );
		}

		removeFromInspection( oldWorld );

		m_worlds.removeAt( wi );
	}
}

void hknpProcessContext::addForInspection( hknpWorld* world )
{
	if ( m_owner && world )
	{
	}
}

void hknpProcessContext::removeFromInspection( hknpWorld* world )
{
	if ( m_owner && world )
	{
	}
}

int hknpProcessContext::findWorld( hknpWorld* world )
{
	return m_worlds.indexOf( world );
}

void hknpProcessContext::addWorldListener( hknpProcessContextListener* cb )
{
	if ( m_addListeners.indexOf(cb) < 0 )
	{
		m_addListeners.pushBack( cb );
	}
}

void hknpProcessContext::removeWorldListener( hknpProcessContextListener* cb )
{
	int index = m_addListeners.indexOf(cb);
	if ( index >= 0 )
	{
		m_addListeners.removeAt( index );
	}
}

void hknpProcessContext::onWorldDestroyedSignal( hknpWorld* world )
{
	removeWorld( world );
}

void hknpProcessContext::setColorScheme( hknpViewerColorScheme* colorScheme )
{
	if( colorScheme )
	{
		m_colorScheme = colorScheme;
	}
	else
	{
		m_colorScheme = &m_defaultColorScheme;
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

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpContactPointViewer.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionData.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

hkpContactPointViewer::hkpContactPointViewer(const hkArray<hkProcessContext*>& contexts, const hkColor::Argb color)
: hkpWorldViewerBase( contexts)
, m_limit(0.0f)
, m_color(color)
{
}

void hkpContactPointViewer::init()
{
	for (int i=0; m_context && i < m_context->getNumWorlds(); ++i)
	{
		worldAddedCallback( m_context->getWorld(i));
	}
}

hkpContactPointViewer::~hkpContactPointViewer()
{
	for (int i=0; m_context && i < m_context->getNumWorlds(); ++i)
	{
		worldRemovedCallback( m_context->getWorld(i));
	}
}

void hkpContactPointViewer::worldAddedCallback( hkpWorld* w)
{
	w->markForWrite();;
	w->addWorldPostSimulationListener(this);
	w->unmarkForWrite();
}

void hkpContactPointViewer::worldRemovedCallback( hkpWorld* w)
{
	w->markForWrite();;
	w->removeWorldPostSimulationListener(this);
	w->unmarkForWrite();
}

static void hkContactPointViewer_displayArrow( hkDebugDisplayHandler* handler, const hkVector4& from, const hkVector4& dir, hkColor::Argb color, int id, int tag  )
{
	HK_TIME_CODE_BLOCK("hkContactPointViewer_displayArrow", HK_NULL);

	// Check that we have a valid direction
	if (dir.lengthSquared<3>().isLess(hkSimdReal_Eps))
	{
		return;
	}

	hkVector4 to; to.setAdd( from, dir );
	hkVector4 ort; hkVector4Util::calculatePerpendicularVector( dir, ort );
	ort.normalize<3>();
	hkVector4 ort2; ort2.setCross( dir, ort );

	ort.mul( dir.length<3>() );

	const hkSimdReal c = hkSimdReal::fromFloat(0.85f);
	hkVector4 p; p.setInterpolate( from, to, c );
	hkVector4 p0; p0.setAddMul( p, ort, hkSimdReal_1 - c );
	hkVector4 p1; p1.setAddMul( p, ort, -(hkSimdReal_1 - c) );

	handler->displayLine( from, to, color, id, tag );
	handler->displayLine( to, p0, color, id, tag );
	handler->displayLine( to, p1, color, id, tag );
}

void hkpContactPointViewer::drawAllContactPointsInIsland(const hkpSimulationIsland* island)
{
	HK_TIME_CODE_BLOCK("hkpContactPointViewer::drawAllContactPointsInIsland", HK_NULL);

	const hkpAgentNnTrack *const tracks[2] = { &island->m_narrowphaseAgentTrack, &island->m_midphaseAgentTrack };

	for ( int i = 0; i < 2; ++i )
	{
		const hkpAgentNnTrack& track = *tracks[i];
		HK_FOR_ALL_AGENT_ENTRIES_BEGIN( track, entry )
		{
			hkpDynamicsContactMgr* manager = static_cast<hkpDynamicsContactMgr*>(entry->m_contactMgr);
			if (manager->getType() == hkpDynamicsContactMgr::TYPE_SIMPLE_CONSTRAINT_CONTACT_MGR)
			{
				hkpSimpleConstraintContactMgr* constaintManager = static_cast<hkpSimpleConstraintContactMgr*>(manager);

				hkLocalArray<hkContactPointId> contactPoints(HK_MAX_CONTACT_POINT);
				constaintManager->getAllContactPointIds(contactPoints);

				for (int c=0; c<contactPoints.getSize(); c++)
				{
					hkContactPoint* contactPoint = constaintManager->getContactPoint(contactPoints[c]);
					
					if (constaintManager->getContactPointProperties(contactPoints[c])->getImpulseApplied() < m_limit) continue;

					const int id = (int)(hkUlong)(constaintManager->m_constraint.getMasterEntity()->getCollidable());
					const hkVector4& pos	= contactPoint->getPosition();
					const hkVector4& normal = contactPoint->getNormal();
					hkContactPointViewer_displayArrow(m_displayHandler, pos, normal, m_color, id, getProcessTag() );
				}
			}
		}
		HK_FOR_ALL_AGENT_ENTRIES_END;
	}
}

void hkpContactPointViewer::postSimulationCallback(hkpWorld* world)
{
	HK_TIME_CODE_BLOCK("hkpActiveContactPointViewer::postSimulationCallback", HK_NULL);

	world->markForRead();

	int i;
	const hkArray<hkpSimulationIsland*>& islands = getIslands(world);

	for (i=0; i<islands.getSize(); i++)
	{
		drawAllContactPointsInIsland(islands[i]);
	}

	world->unmarkForRead();
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

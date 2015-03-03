/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpToiContactPointViewer.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Collide/Agent3/Machine/Nn/hkpAgentNnMachine.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/Collide/hkpSimpleConstraintContactMgr.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionData.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>


int hkpToiContactPointViewer::s_tag = 0;

void HK_CALL hkpToiContactPointViewer::registerViewer()
{
	s_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkProcess* HK_CALL hkpToiContactPointViewer::create(const hkArray<hkProcessContext*>& contexts)
{
	return new hkpToiContactPointViewer(contexts);
}


int hkpToiContactPointViewer::getProcessTag(void)
{
	return s_tag;
}

hkpToiContactPointViewer::hkpToiContactPointViewer(const hkArray<hkProcessContext*>& contexts)
: hkpWorldViewerBase(contexts)
{
}

void hkpToiContactPointViewer::init()
{
	for (int i=0; m_context && i < m_context->getNumWorlds(); ++i)
	{
		worldAddedCallback( m_context->getWorld(i));
	}
}

hkpToiContactPointViewer::~hkpToiContactPointViewer()
{
	for (int i=0; m_context && i < m_context->getNumWorlds(); ++i)
	{
		worldRemovedCallback( m_context->getWorld(i));
	}
}

void hkpToiContactPointViewer::worldAddedCallback( hkpWorld* w)
{
	w->markForWrite();;
	w->addContactListener(this);
	w->unmarkForWrite();
}

void hkpToiContactPointViewer::worldRemovedCallback( hkpWorld* w)
{
	w->markForWrite();;
	w->removeContactListener(this);
	w->unmarkForWrite();
}

static void hkToiContactPointViewer_displayArrow( hkDebugDisplayHandler* handler, const hkVector4& from, const hkVector4& dir, hkColor::Argb color, int id, int tag  )
{
	// Check that we have a valid direction
	if (dir.lengthSquared<3>().isGreater(hkSimdReal_Eps))
	{
		HK_TIME_CODE_BLOCK("ToiDisplayArrow", HK_NULL);

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
}


void hkpToiContactPointViewer::contactPointCallback( const hkpContactPointEvent& event )
{
	if ( event.isToi() )
	{
		const int id = (int)(hkUlong)( event.m_bodies[0]->getCollidable() ); // hmm, lets just choose the first one...
		hkToiContactPointViewer_displayArrow( m_displayHandler, event.m_contactPoint->getPosition(), event.m_contactPoint->getNormal(), hkColor::RED, id, s_tag );
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

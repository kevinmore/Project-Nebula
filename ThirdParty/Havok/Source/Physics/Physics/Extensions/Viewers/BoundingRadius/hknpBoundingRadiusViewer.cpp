/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/BoundingRadius/hknpBoundingRadiusViewer.h>

#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/Shape/hkDisplaySphere.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include "Common/Visualize/Shape/hkDisplayConvex.h"


int hknpBoundingRadiusViewer::s_tag = 0;

void HK_CALL hknpBoundingRadiusViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpBoundingRadiusViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hknpBoundingRadiusViewer( contexts );
}


hknpBoundingRadiusViewer::hknpBoundingRadiusViewer( const hkArray<hkProcessContext*>& contexts )
: hknpViewer( contexts )
{
}

void hknpBoundingRadiusViewer::step( hkReal deltaTime )
{
	if( !m_context )
	{
		return;
	}

	HK_TIMER_BEGIN( "BoundingRadiusViewer", this );

	for( int wi=0; wi<m_context->getNumWorlds(); wi++ )
	{
		hknpWorld* world = m_context->getWorld(wi);
		for( hknpBodyIterator it = world->getBodyIterator(); it.isValid(); it.next() )
		{
			const hknpBody& body = it.getBody();
			if ( body.isAddedToWorld() && body.isDynamic() )
			{
				const hknpMotion& motion = world->getMotion( body.m_motionId );

				hkSimdReal boundingRadius; boundingRadius.setFromHalf( body.m_radiusOfComCenteredBoundingSphere );
				hkDisplaySphere displaySphere( hkSphere( motion.getCenterOfMassInWorld(), boundingRadius ), 1, 1 );
				displaySphere.buildGeometry();

				// HACK: hkDisplaySphere is not drawn correctly by VDB, so convert to hkDisplayConvex
				hkDisplayConvex displayConvex( displaySphere.getGeometry() );
				displayConvex.setTransform( displaySphere.getTransform() );

				hkArray<hkDisplayGeometry*> displayGeometries( 1, &displayConvex );
				m_displayHandler->displayGeometry( displayGeometries, hkColor::MAGENTA, 0, s_tag );

				// HACK: Avoid hkDisplayConvex trying to delete the geometry
				new (&displayConvex) hkDisplayConvex(HK_NULL);
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

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/Cell/hknpCellViewer.h>

#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/Shape/hkDisplaySphere.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#include <Common/Visualize/Shape/hkDisplayConvex.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/Shape/hkDisplayAABB.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Physics/Internal/Dynamics/World/SpaceSplitter/hknpSpaceSplitter.h>

#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>
#include <Physics/Physics/Dynamics/World/ModifierManager/hknpModifierManager.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>

#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>


int hknpCellViewer::s_tag = 0;

void HK_CALL hknpCellViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpCellViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hknpCellViewer( contexts );
}


hknpCellViewer::hknpCellViewer( const hkArray<hkProcessContext*>& contexts )
: hknpViewer( contexts )
{
}

void hknpCellViewer::step( hkReal deltaTime )
{
	if( !m_context )
	{
		return;
	}

	HK_TIMER_BEGIN( "CellViewer", this );

	for( int wi=0; wi<m_context->getNumWorlds(); wi++ )
	{
		hknpWorld* world = m_context->getWorld(wi);

		const hkArray<hknpBodyId>& activeBodies = world->getActiveBodies();
		for (int bi = 0; bi < activeBodies.getSize(); bi++)
		{
			hknpBodyId id = activeBodies[bi];
			const hknpBody& body = world->getBody(id);
			hkAabb aabb; world->m_intSpaceUtil.restoreAabb( body.m_aabb, aabb);

			const hknpMotion& motion = world->getMotion(body.m_motionId);
			hknpCellIndex cellIndex = motion.m_cellIndex;

			hkDisplayAABB displayAabb;	displayAabb.setExtents( aabb.m_min, aabb.m_max);
			hkInplaceArray<hkDisplayGeometry*,1> displayGeometries;	displayGeometries.pushBackUnchecked(&displayAabb);
			hkColor::Argb color = hkColor::rgbFromFloats((cellIndex%5)*0.25f, 1.0f-(cellIndex%3)*0.5f, (cellIndex%2)*1.0f);
			m_displayHandler->displayGeometry(displayGeometries, hkTransform::getIdentity(), color, 0, s_tag);
		}

		// show the bodyId for the viewer specific body
		if( world == m_worldForViewerSpecificBody && m_selectedBody != hknpBodyId(0) )
		{
			const hknpBody& body = world->getBody(m_selectedBody);

			const hkVector4& bodyPosition = body.getTransform().getTranslation();

			hkStringBuf bodyText;
			bodyText.printf("BodyId: %i \n", m_selectedBody.value());
			m_displayHandler->display3dText(bodyText.cString(), bodyPosition, 0xFFFFFFFF, 0, s_tag);
		}
	}

	HK_TIMER_END();
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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

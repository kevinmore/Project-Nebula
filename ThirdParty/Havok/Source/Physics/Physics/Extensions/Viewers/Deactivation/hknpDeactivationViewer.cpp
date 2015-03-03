/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/Deactivation/hknpDeactivationViewer.h>

#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/Shape/hkDisplayAABB.h>
#include <Common/Base/Types/Color/hkColor.h>

#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationManager.h>
#include <Physics/Internal/Dynamics/World/Deactivation/hknpDeactivationState.h>


int hknpDeactivationViewer::s_tag = 0;

void HK_CALL hknpDeactivationViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpDeactivationViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hknpDeactivationViewer( contexts );
}


hknpDeactivationViewer::hknpDeactivationViewer( const hkArray<hkProcessContext*>& contexts )
	: hknpViewer( contexts )
{
}

void hknpDeactivationViewer::step( hkReal deltaTime )
{
	if ( !m_context )
	{
		return;
	}

	HK_TIMER_BEGIN( "DeactivationViewer", this );

	for ( int wi = 0; wi < m_context->getNumWorlds(); wi++ )
	{
		hknpWorld* world = m_context->getWorld(wi);

		// Active Bodies
		{
			const hkArray<hknpBodyId>& activeBodyIds = world->getActiveBodies();
			const hkUint32 numActiveBodies = activeBodyIds.getSize();

			hkArray<hkDisplayAABB>::Temp displayAabbs;
			displayAabbs.setSize( numActiveBodies );

			hkArray<hkDisplayGeometry*>::Temp activeGeometries;
			activeGeometries.setSize( numActiveBodies );

			for (hkUint32 activeBodyIdi = 0; activeBodyIdi < numActiveBodies; activeBodyIdi++)
			{
				hknpBodyId activeBodyId = activeBodyIds[activeBodyIdi];
				const hknpBody& body = world->getBody(activeBodyId);

				hkAabb aabb; world->m_intSpaceUtil.restoreAabb( body.m_aabb, aabb );
				hkDisplayAABB& displayAabb = displayAabbs[activeBodyIdi];
				displayAabb.setExtents( aabb.m_min, aabb.m_max );

				activeGeometries[activeBodyIdi] = &displayAabb;
			}

			m_displayHandler->displayGeometry( activeGeometries, hkColor::BLUE, 0, s_tag );
		}

		// Inactive Islands
		{
			const hkArray<hknpDeactivatedIsland*> &inactiveIslands = world->m_deactivationManager->m_deactivatedIslands;
			const hkUint32 numInactiveIslands = inactiveIslands.getSize();

			hkArray<hkDisplayAABB>::Temp displayAabbs;
			displayAabbs.setSize( numInactiveIslands );

			hkArray<hkDisplayGeometry*>::Temp inactiveIslandGeometries;
			inactiveIslandGeometries.reserve( numInactiveIslands );

			for (hkUint32 inactiveIslandi = 0; inactiveIslandi < numInactiveIslands; inactiveIslandi++)
			{
				hknpDeactivatedIsland* inactiveIsland = inactiveIslands[inactiveIslandi];
				if(inactiveIsland)
				{
					hkAabb inactiveIslandAabb; inactiveIslandAabb.setEmpty();
					const hkArray<hknpBodyId> &bodyIds = inactiveIsland->m_bodyIds;
					for (int bodyIdi = 0; bodyIdi < bodyIds.getSize(); bodyIdi++)
					{
						hknpBodyId bodyId = bodyIds[bodyIdi];
						const hknpBody& body = world->getBody(bodyId);
						hkAabb aabb; world->m_intSpaceUtil.restoreAabb( body.m_aabb, aabb );
						inactiveIslandAabb.includeAabb(aabb);
					}
					hkDisplayAABB& displayAabb = displayAabbs[inactiveIslandi];
					displayAabb.setExtents(inactiveIslandAabb.m_min, inactiveIslandAabb.m_max);

					inactiveIslandGeometries.pushBackUnchecked( &displayAabb );
				}
			}

			m_displayHandler->displayGeometry( inactiveIslandGeometries, hkColor::GREEN, 0, s_tag );
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

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Extensions/Viewers/CompositeQueryAabb/hknpCompositeQueryAabbViewer.h>

#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/hkProcessFactory.h>

#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>


int hknpCompositeQueryAabbViewer::s_tag = 0;

void HK_CALL hknpCompositeQueryAabbViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpCompositeQueryAabbViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hknpCompositeQueryAabbViewer( contexts );
}


hknpCompositeQueryAabbViewer::hknpCompositeQueryAabbViewer( const hkArray<hkProcessContext*>& contexts )
:	hknpViewer( contexts )
{
}


void hknpCompositeQueryAabbViewer::step( hkReal deltaTime )
{

}


int hknpCompositeQueryAabbViewer::getEnabledFunctions()
{
	return (1<<FUNCTION_POST_COMPOSITE_QUERY_AABB);
}

int hknpCompositeQueryAabbViewer::postCompositeQueryAabb(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpAabbQuery& aabbQuery, const hknpShapeQueryInfo& queryShapeInfo, const hknpShapeQueryInfo& targetShapeInfo,
	hknpShapeKey* keys, int numKeys, int maxNumKeys )
{
	HK_DISPLAY_TRANSFORMED_BOUNDING_BOX(*targetShapeInfo.m_shapeToWorld, aabbQuery.m_aabb, hkColor::GREEN);
	for (int i=0; i<numKeys; i++)
	{
		hknpShapeCollectorWithInplaceTriangle leafShapeCollector;
		//leafShapeCollector.reset( *targetShapeInfo.m_shapeToWorld );
		leafShapeCollector.reset( hkTransform::getIdentity() );
		targetShapeInfo.m_rootShape->getLeafShape(keys[i], &leafShapeCollector);
		hkAabb leafAabb;
		leafShapeCollector.m_shapeOut->calcAabb(hkTransform::getIdentity(), leafAabb);
		HK_DISPLAY_TRANSFORMED_BOUNDING_BOX(*targetShapeInfo.m_shapeToWorld, leafAabb, hkColor::semiTransparent( hkColor::BLUE ));
	}
	return numKeys;
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

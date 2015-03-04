/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/Filter/DisableCollision/hknpDisableCollisionFilter.h>

hknpDisableCollisionFilter	hknpDisableCollisionFilter::g_instance;


hknpDisableCollisionFilter::hknpDisableCollisionFilter()
:	hknpCollisionFilter( hknpCollisionFilter::ALWAYS_HIT_FILTER )
{
}


hknpDisableCollisionFilter* hknpDisableCollisionFilter::getInstancePtr()
{
	return &g_instance;
}


#if !defined(HK_PLATFORM_SPU)
int hknpDisableCollisionFilter::filterBodyPairs( const hknpSimulationThreadContext& context, hknpBodyIdPair* pairs, int numPairs ) const
{
	return 0;
}
#endif


bool hknpDisableCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	hknpBroadPhaseLayerIndex layerIndex ) const
{
	// Never collide.
	return false;
}


bool hknpDisableCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	hknpBodyId bodyIdA,
	hknpBodyId bodyIdB ) const
{
	// Never collide.
	return false;
}


bool hknpDisableCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	const hknpQueryFilterData& queryFilterData,
	const hknpBody& body ) const
{
	// Never collide.
	return false;
}


bool hknpDisableCollisionFilter::isCollisionEnabled(
	hknpCollisionQueryType::Enum queryType,
	bool targetShapeIsB,
	const FilterInput& shapeInputA,
	const FilterInput& shapeInputB ) const
{
	// Never collide.
	return false;
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

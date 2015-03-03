/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_DISABLE_COLLISION_FILTER_H
#define HKNP_DISABLE_COLLISION_FILTER_H

#include <Physics/Physics/Collide/Filter/hknpCollisionFilter.h>

extern const hkClass hknpDisableCollisionFilterClass;

/// This filter will not report any collision.
class hknpDisableCollisionFilter : public hknpCollisionFilter
{
	public:

		HK_DECLARE_REFLECTION();
		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor.
		hknpDisableCollisionFilter();

		//
		// hknpCollisionFilter implementation
		//

#if !defined( HK_PLATFORM_SPU )

		virtual int filterBodyPairs(
			const hknpSimulationThreadContext& context,
			hknpBodyIdPair* pairs, int numPairs ) const HK_OVERRIDE;

#endif

		virtual bool isCollisionEnabled(
			hknpCollisionQueryType::Enum queryType,
			hknpBroadPhaseLayerIndex layerIndex ) const HK_OVERRIDE;

		virtual bool isCollisionEnabled(
			hknpCollisionQueryType::Enum queryType,
			hknpBodyId bodyIdA, hknpBodyId bodyIdB ) const HK_OVERRIDE;

		virtual bool isCollisionEnabled(
			hknpCollisionQueryType::Enum queryType,
			const hknpQueryFilterData& queryFilterData, const hknpBody& body ) const HK_OVERRIDE;

		virtual bool isCollisionEnabled(
			hknpCollisionQueryType::Enum queryType,
			bool targetShapeIsB, const FilterInput& shapeInputA, const FilterInput& shapeInputB ) const HK_OVERRIDE;

	public:

		static hknpDisableCollisionFilter* getInstancePtr();
		static hknpDisableCollisionFilter g_instance;

		hknpDisableCollisionFilter( hkFinishLoadedObjectFlag flag ) : hknpCollisionFilter( flag ) {}
};


#endif // HKNP_DISABLE_COLLISION_FILTER_H

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

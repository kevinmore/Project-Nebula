/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_GROUP_FILTER_H
#define HKNP_GROUP_FILTER_H

#include <Physics/Physics/Collide/Filter/hknpCollisionFilter.h>

extern const hkClass hknpGroupCollisionFilterClass;


class hknpGroupCollisionFilter : public hknpCollisionFilter
{
	public:

		enum { COLLISION_LUT_SIZE = 32 };

	public:

		/// Combine the layer and systemGroup information into one 32 bit integer.
		static HK_FORCE_INLINE hkUint32 HK_CALL calcFilterInfo(
			int layer, int systemGroup = 0, int subSystemId = 0, int subSystemDontCollideWith = 0 );

		/// Extract the layer from a given \a filterInfo.
		static HK_FORCE_INLINE int HK_CALL getLayerFromFilterInfo( hkUint32 filterInfo );

		/// Returns the provided \a filterInfo where the layer has been replaced by \a newLayer.
		static HK_FORCE_INLINE int HK_CALL setLayer( hkUint32 filterInfo, int newLayer );

		/// Extract the system group from a given \a filterInfo.
		static HK_FORCE_INLINE int HK_CALL getSystemGroupFromFilterInfo( hkUint32 filterInfo );

		/// Extract the subsystem id from a given \a filterInfo.
		static HK_FORCE_INLINE int HK_CALL getSubSystemIdFromFilterInfo( hkUint32 filterInfo );

		/// Extract the subSystemDontCollideWith from a given \a filterInfo.
		static HK_FORCE_INLINE int HK_CALL getSubSystemDontCollideWithFromFilterInfo( hkUint32 filterInfo );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor; enables all collisions between all layers by default.
		hknpGroupCollisionFilter();

		/// Enable the collision between the layers A and B.
		void enableCollisionsBetween( int layerA, int layerB );

		/// Disables collisions between the layers A and B.
		void disableCollisionsBetween( int layerA, int layerB );

		/// Enables collisions between the specified layers.
		/// \a layerBitsA and \a layerBitsB are bitfields, one bit for every layer.
		/// To enable collisions between e.g. one layer and all other layers,
		/// call enableCollisionsUsingBitfield( 1<<myLayer, 0xfffffffe)
		void enableCollisionsUsingBitfield( hkUint32 layerBitsA, hkUint32 layerBitsB );

		/// Disables collisions between the specified collision layers.
		/// See enableCollisionsUsingBitfield for details on the usage of layerBits bitfields.
		void disableCollisionsUsingBitfield( hkUint32 layerBitsA, hkUint32 layerBitsB );

		/// Check
		HK_FORCE_INLINE hkBool32 _isCollisionEnabled( hkUint32 infoA, hkUint32 infoB ) const;

		/// Creates a new unique identifier for system groups (maximum 65k)
		HK_FORCE_INLINE int getNewSystemGroup();

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
			bool targetShapeIsB, const FilterInput& shapeInputA, const FilterInput& shapeInputB ) const;

	public:

		// This member must remain the first member in hknpGroupCollisionFilter and needs to be 16-byte aligned for SPU
		HK_ALIGN16( int m_nextFreeSystemGroup );

		hkUint32 m_collisionLookupTable[COLLISION_LUT_SIZE];

		hknpGroupCollisionFilter( class hkFinishLoadedObjectFlag flag ) : hknpCollisionFilter(flag) {}
};


#include <Physics/Physics/Collide/Filter/Group/hknpGroupCollisionFilter.inl>


#endif // HKNP_GROUP_FILTER_H

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

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_PAIR_COLLISION_FILTER_H
#define HKNP_PAIR_COLLISION_FILTER_H

#include <Physics/Physics/Collide/Filter/hknpCollisionFilter.h>


/// A simple pairwise body collision filter.
///
/// This filter works at BODY level, i.e. you cannot use it to enable/disable collisions for shapes.
/// You can specify a child filter which will be executed first.
/// For all SHAPE level callbacks this filter will either return its child filter's result (if available) or otherwise
/// TRUE.
class hknpPairCollisionFilter : public hknpCollisionFilter
{
	public:

		HK_CLASSALIGN(struct,4) Key
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_BASE, Key );
			hknpBodyId m_bodyIdA;
			hknpBodyId m_bodyIdB;
		};

		struct MapOperations
		{
			inline static unsigned hash( Key key, unsigned modulus )
			{
				// using Knuth's multiplicative golden hash
				hkUint32 keyA = key.m_bodyIdA.value();
				hkUint32 keyB = key.m_bodyIdB.value();
				hkUint64 combinedKey = keyB + ( (hkUint64)keyA << 32 );
				return unsigned( ( ( combinedKey * 2654435761U ) & modulus ) & 0xffffffff );
			}
			inline static void invalidate( Key& key )
			{
				key.m_bodyIdA = hknpBodyId::invalid();
				key.m_bodyIdB = hknpBodyId::invalid();
			}
			inline static hkBool32 isValid( Key key )
			{
				return ( key.m_bodyIdA.isValid() && key.m_bodyIdB.isValid() );
			}
			inline static hkBool32 equal( Key key0, Key key1 )
			{
				return ( ( key0.m_bodyIdA == key1.m_bodyIdA ) && ( key0.m_bodyIdB == key1.m_bodyIdB ) );
			}
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		HK_DECLARE_REFLECTION();

		/// Constructor. The \a childFilter (if set to != HK_NULL) will be executed prior to this pair filter.
		hknpPairCollisionFilter( const hknpCollisionFilter* childFilter );

		/// Serialization constructor.
		hknpPairCollisionFilter( hkFinishLoadedObjectFlag flag );

		/// Disable collisions between body A and body B. Returns the number of times the pair is now 'disabled'.
		int disableCollisionsBetween( hknpWorld* world, hknpBodyId bodyIdA, hknpBodyId bodyIdB );

		/// Enable collisions between body A and body B. Returns the number of times the pair is still 'disabled'.
		int enableCollisionsBetween( hknpWorld* world, hknpBodyId bodyIdA, hknpBodyId bodyIdB );

		/// Forces all collisions that have previously been disabled to get enabled again, independent of their 'count'.
		void clearAll();

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

	protected:

		// Builds the map key from the two body ids.
		HK_FORCE_INLINE void calcKey( hknpBodyId bodyIdA, hknpBodyId bodyIdB, Key& keyOut ) const;

		// Checks if the two supplied bodies are allowed to collide.
		HK_FORCE_INLINE bool _isCollisionEnabled( hknpBodyId bodyIdA, hknpBodyId bodyIdB ) const;

	public:

		// Workaround helper class for serialization
		class MapPairFilterKeyOverrideType
		{
			public:

				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DYNAMICS, hknpPairCollisionFilter::MapPairFilterKeyOverrideType );
				HK_DECLARE_REFLECTION();

				// Members match those in hkMapBase
				void* m_elem; //+nosave
				int m_numElems;
				int m_hashMod;
		};

		class hkMap<Key, hkUint32, MapOperations> m_disabledPairs; //+nosave +overridetype(class hknpPairCollisionFilter::MapPairFilterKeyOverrideType)

		const hknpCollisionFilter* m_childFilter;
};


#include <Physics/Physics/Collide/Filter/Pair/hknpPairCollisionFilter.inl>


#endif // HKNP_PAIR_COLLISION_FILTER_H

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

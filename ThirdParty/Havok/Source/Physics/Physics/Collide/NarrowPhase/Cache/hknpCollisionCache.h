/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_COLLISION_CACHE_H
#define HKNP_COLLISION_CACHE_H

#include <Physics/Physics/hknpTypes.h>
#include <Common/Base/Container/BlockStream/hkBlockStream.h>
#include <Common/Base/Math/Vector/hkPackedVector3.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Geometry/Internal/Algorithms/Gsk/hkcdGskData.h>
#include <Physics/Physics/Dynamics/Body/hknpBodyQuality.h>
#include <Physics/Physics/Collide/Query/hknpQueryAabbNmpUtil.h>
#include <Physics/Physics/Dynamics/Body/hknpBody.h>

struct hknpCollisionCache;
struct hknpCdBodyBase;

class hknpCdCacheStream:    public hkBlockStream<hknpCollisionCache> {};
class hknpCdCacheReader:	public hkBlockStream<hknpCollisionCache>::Reader {};
class hknpCdCacheConsumer:	public hkBlockStream<hknpCollisionCache>::Consumer {};
class hknpCdCacheModifier:  public hkBlockStream<hknpCollisionCache>::Modifier {};
class hknpCdCacheWriter:	public hkBlockStream<hknpCollisionCache>::Writer {};

#if defined(HK_PLATFORM_WIN32)
#	pragma warning( 3 : 4820 )		
#endif


/// Base class of all collision caches.
struct hknpCollisionCache	// size: 16 bytes
{
	public:

		/// Get the total size of the cache in bytes.
		HK_FORCE_INLINE int getSizeInBytes() const { return m_sizeInQuads << 4; }

		/// Destroys the object.
		/// Does not set values to zero. Call reset() to continue using this cache.
		/// This fires callbacks if the cache is of type hknpManifoldCollisionCache.
		HK_FORCE_INLINE void destruct(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,
			hknpCdBodyBase* HK_RESTRICT cdBodyA, hknpCdBodyBase* HK_RESTRICT cdBodyB,
			hknpCdCacheDestructReason::Enum reason );

		/// Move the child caches of this stream to childCdCacheWriter.
		HK_FORCE_INLINE void moveAndConsumeChildCaches(
			const hknpSimulationThreadContext& tl,
			hknpCdCacheStream* childCdCacheStreamIn,
			hknpCdCacheStream* childCdCacheStreamInPpu,
			hknpCdCacheWriter* childCdCacheWriter );

#if !defined(HK_PLATFORM_SPU)
		/// Copy the child caches (does not consume)
		HK_FORCE_INLINE void moveChildCachesWithoutConsuming(
			const hknpSimulationThreadContext& tl,
			hknpCdCacheWriter* childCdCacheWriter );
#endif

		/// Retrieve the shape keys, depending on the cache type, either shapeKeyA or shapeKeyB or none are set.
		/// This can be used when iterating over child caches and to set the shapeKeyA and shapeKeyB.
		/// If you want to destruct child caches, you should use this function to set the m_shapeKey
		/// on the hknpCdBody.
		HK_FORCE_INLINE void getLeafShapeKeys(
			HK_PAD_ON_SPU(hknpShapeKey)* shapeKeyA, HK_PAD_ON_SPU(hknpShapeKey)* shapeKeyB ) const;

		/// Set the shape key (if bodies are not used).
		HK_FORCE_INLINE void setShapeKey( hknpShapeKey key ) { *shapeKeyPtr() = key; }

		/// Get the shape key (if bodies are not used).
		HK_FORCE_INLINE hknpShapeKey getShapeKey() const { return *shapeKeyPtr(); }

		/// Get the body ID pair.
		HK_FORCE_INLINE const hknpBodyIdPair& getBodyIdPair() const { return *bodyIdPairPtr(); }

	protected:

		/// Initialize a cache. Does not set m_bodyA, m_bodyB
		HK_FORCE_INLINE void init( hknpCollisionCacheType::Enum type, int size );

#if HK_ENDIAN_BIG
		HK_FORCE_INLINE hknpShapeKey* shapeKeyPtr()	{ return (hknpShapeKey*)&m_bodyB; }
		HK_FORCE_INLINE const hknpShapeKey* shapeKeyPtr() const	{ return (hknpShapeKey*)&m_bodyB; }
		HK_FORCE_INLINE const hknpBodyIdPair* bodyIdPairPtr() const { return (hknpBodyIdPair*)&m_bodyB; }
#else
		HK_FORCE_INLINE hknpShapeKey* shapeKeyPtr()	{ return (hknpShapeKey*)&m_bodyA; }
		HK_FORCE_INLINE const hknpShapeKey* shapeKeyPtr() const	{ return (hknpShapeKey*)&m_bodyA; }
		HK_FORCE_INLINE const hknpBodyIdPair* bodyIdPairPtr() const { return (hknpBodyIdPair*)&m_bodyA; }
#endif

	public:

		// The following are sorted by architecture so that they can be treated as a hknpBodyIdPair for sorting,
		// with bodyB having higher priority.
		// They can also be interpreted as a shape key for some caches (see getShapeKey() ).
#if HK_ENDIAN_BIG	// PPC, ARB
		hknpBodyId m_bodyB;										// 4 bytes
		hknpBodyId m_bodyA;										// 4 bytes
#else				// INTEL
		hknpBodyId m_bodyA;										// 4 bytes
		hknpBodyId m_bodyB;										// 4 bytes
#endif

		/// The type of cache
		hkEnum<hknpCollisionCacheType::Enum, hkUint8> m_type;	// 1 byte

		/// Size of the cache in quads
		hkUint8 m_sizeInQuads;									// 1 byte

		/// A measure of the relative distance by which the bodies must move
		/// before their collision detection results will change.
		hknpLinearTimType m_linearTim;							// 2 bytes

		/// Flags detailing SPU support (bodyA.m_spuFlags | bodyB.m_spuFlags)
		hknpBody::SpuFlags m_spuFlags;							// 1 byte

		HKNP_PADDING_BYTES(m_padding0,3);						// 3 bytes
};


// ---------------------------------------------------------------------------------------------------------------------
// Convex caches
// ---------------------------------------------------------------------------------------------------------------------

/// Base cache for convex convex collisions.
/// Creation of this cache is silent, no callbacks are fired at all. As soon as real contact points are generated,
/// this cache is converted to a hknpManifoldCollisionCache and hknpModifier::manifoldCreatedCallback() is fired.
struct hknpConvexConvexCollisionCache : public hknpCollisionCache	// size: 32 bytes
{
	public:

		/// The maximum number of bytes for all properties combined.
		enum { MAX_PROPERTY_BUFFER_SIZE = 64 };

	public:

#if !defined(HK_PLATFORM_SPU)

		/// Construct a convex-convex collision cache in the given cache writer.
		static void HK_CALL construct(
			const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB, hknpCdCacheWriter* cacheWriterInOut );

#endif

	public:

		/// Clear all values.
		HK_FORCE_INLINE void init();

		/// This is called by derived classes, if this cache (now a hknpConvexConvexCollisionCache) becomes a cache of type T.
		template<class T>
		HK_FORCE_INLINE T* promoteTo( int sizeOfT = sizeof(T) );

		/// Inline version of destructCdCacheImpl.
		HK_FORCE_INLINE void _destructCdCacheImpl(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB, hknpCdCacheDestructReason::Enum reason ) const;

		/// See hknpCollisionCache::reset()
		HK_FORCE_INLINE void resetCdCacheImpl();

		/// Returns true if you can cast this class to hknpManifoldCollisionCache
		HK_FORCE_INLINE bool hasManifoldData() const;

		/// Implementation of hknpCollisionCache::getLeafShapeKeys().
		HK_FORCE_INLINE void getLeafShapeKeysImpl(
			HK_PAD_ON_SPU(hknpShapeKey)* shapeKeyA, HK_PAD_ON_SPU(hknpShapeKey)* shapeKeyB ) const;

		/// Set the cache quality based on the given inputs.
		HK_FORCE_INLINE void setQuality( const hknpSolverInfo& solverInfo,
			const hknpBody& bodyA, const hknpBodyQuality* qualityA, const hknpMaterial& materialA, const hknpShape& shapeA,
			const hknpBody& bodyB, const hknpBodyQuality* qualityB, const hknpMaterial& materialB, const hknpShape& shapeB );

	protected:

		// Constructor is never called. Call construct() instead.
		hknpConvexConvexCollisionCache() {}

		// Get the offset (in 4-bytes units) from the start of the property buffer to the property with the given key.
		hkUint8 getPropertyOffset( hkUint8 key ) const;

		// Set the offset (in 4-bytes units) from the start of the property buffer to the property with the given key.
		void setPropertyOffset( hkUint8 key, hkUint8 offset );

	public:

		/// A cache used during GSK collision detection
		HK_ALIGN( hkcdGsk::Cache m_gskCache, 4 );						// 5 bytes

		/// Used for marking edges of triangles (and quads) for welding. Only lower 4 bits are used (one bit per edge)
		hkUint8 m_edgeWeldingInfo;										// 1 byte

		/// A combination of the quality flags from both bodies
		hkFlags<hknpBodyQuality::FlagsEnum, hkUint16> m_qualityFlags;	// 2 bytes

	protected:

		/// Contains the offsets of any allocated properties. Every half-byte contains an offset (in 4-bytes units)
		/// from the start of the property buffer to the property data for that key.
		/// This is used in hknpManifoldCollisionCache, but stored here in order to pack memory.
		hkUint32 m_propertyOffsets;										// 4 bytes

		HKNP_PADDING_BYTES(m_padding1, 4);								// 4 bytes
};


/// The common base cache of all narrow phase collision caches if contact points are present.
///
/// As soon as the first real contact points is generated in hknpConvexConvexCollisionCache, the
/// hknpConvexConvexCollisionCache cache is converted to a hknpManifoldCollisionCache and
/// hknpModifier::manifoldCreatedCallback() is fired.
struct hknpManifoldCollisionCache : public hknpConvexConvexCollisionCache
{
	public:

		/// Keys used to allocate and access optional cache properties.
		enum PropertyKey
		{
			BUFFER_SIZE_PROPERTY = 0,	/// Special property storing the size of the properties buffer (not an offset)
			FIRST_PROPERTY_KEY,

			// Internal properties
			RESTITUTION_PROPERTY = FIRST_PROPERTY_KEY,	/// Data used by the restitution modifier
			SOFT_CONTACT_PROPERTY,						/// Data used by the soft contact modifier
			RESERVED_PROPERTY,							/// Reserved for future use

			// User properties
			USER_PROPERTY_0,
			USER_PROPERTY_1,
			USER_PROPERTY_2,
			USER_PROPERTY_3,

			NUM_PROPERTY_KEYS
		};

	public:

		/// Constructor
		hknpManifoldCollisionCache() : hknpConvexConvexCollisionCache(), m_propertyKeysUsed(0) {}

		/// Set the friction and restitution from the combined input materials.
		void setFrictionAndRestitution( const hknpMaterial& x, const hknpMaterial& y );

		//
		// Destruction
		//

		/// Call this if you want to destroy a hknpManifoldCollisionCache.
		/// Note that the hknpCollisionCache has no destructCdCacheImpl methods and can be silently deleted
		void fireManifoldDestroyed(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
			hknpCdCacheDestructReason::Enum reason );

		/// Inline version of above function.
		HK_FORCE_INLINE void _fireManifoldDestroyed(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
			hknpCdCacheDestructReason::Enum reason );

		/// converts caches size to be of type hknpConvexConvexCollisionCache
		HK_FORCE_INLINE void demoteToCvxCvxCache();

		//
		// Properties
		//

		/// Allocates enough memory to store an object of type T for the given property key, with the requested alignment.
		/// Returns a pointer to the uninitialized allocated memory on success, or asserts and returns HK_NULL on failure.
		/// Notes:
		///   - you can only call this function from hknpModifier::manifoldCreatedCallback() implementations.
		///   - alignment must be a power of two. The minimum (and default) alignment value is 4-bytes.
		template < typename T >
		HK_FORCE_INLINE T* allocateProperty( PropertyKey key, hkUint32 alignment = 4 );

		/// Returns a pointer to the corresponding property, or HK_NULL if the property has not been allocated.
		template < typename T >
		T* accessProperty( PropertyKey key ) const;

		/// Returns the total size of the property buffer.
		HK_FORCE_INLINE hkUint8 getPropertyBufferSize() const;

	public:

		/// Solver results from the previous step.
		HK_ALIGN16( hknpManifoldSolverInfo m_manifoldSolverInfo );	// aligned because of SPU access

		/// body[0/1].m_flags | material[0/1].m_flags
		hknpBodyFlags m_bodyAndMaterialFlags;

		hkHalf m_friction;				///< Dynamic friction
		hkHalf m_staticFrictionExtra;	///< The extra amount of friction for static
		hkHalf m_restitution;			///< Restitution

		/// Maximum impulse which can be applied by any of the contact points during each solver sub step.
		/// Impulses larger than this will be clipped by the solver.
		hkHalf m_maxImpulse;

		/// A fraction of the max contact impulse to apply when the impulse is clipped.
		/// This is compressed from [0.0f, 1.0f] to [0, 255].
		/// If set to 255, an impulse of m_maxContactImpulse will be applied. (default)
		/// If set to 0, no impulse will be applied.
		hkUint8 m_fractionOfClippedImpulseToApply;

		hkUint8 m_numContactPoints;		///< Number of contact points in the manifold.

		hkUint8 m_propertyBufferOffset;	///< Offset to the start of the property buffer. This is the size of the derived class.

		/// Flags to mark which property keys have been set in this cache.
		/// Bits 1-7 correspond to the keys, and the first bit indicates whether the buffer is full.
		hkUint8 m_propertyKeysUsed;

#if HK_POINTER_SIZE == 8
		HKNP_PADDING_BYTES( m_padding2, 8 );
#endif

		hkVector4 m_integratedFrictionRhs;	///< Internal variable to help with friction.

		
		hkVector4 m_distanceOffset;		///< Internal variable to help with contact solving.
		hkVector4 m_prevFrameRhs;		///< Internal variable to help with contact solving.
};


/// A collision cache for convex convex interactions.
/// The lifetime of this cache can be tracked through modifiers (see fireXxx methods).
struct hknpConvexConvexManifoldCollisionCache : public hknpManifoldCollisionCache
{
	private:

		/// Constructor is never called, only the base hknpCollisionCache constructor is called.
		/// hknpCollisionCache is converted to a hknpConvexConvexManifoldCollisionCache by calling promoteTo<>().
		hknpConvexConvexManifoldCollisionCache(); // : hknpManifoldCollisionCache()

	public:

		HK_ALIGN16(hkPackedUnitVector<3> m_normal);	// Normal in world space. Aligned for faster decompression. (size = 6 bytes)
		hkUint16	m_faceIndices;		///< Supporting faces indices. if set to ~0 the cache (including m_bestQuad) is uninitialized
		hkUint8		m_bestQuad[4];		///< Manifold quad.
		hkUint8		m_numPoints;		///< The number of contact points stored in m_bestQuad
		hkUint8		m_angularTim;		///< The angular TIM before recalling GSK to find the best facing plane

		hkUint8		m_faceVsQuadMask;	// one bit for each vertex on a face for hknpCollision2DFastProcessUtil::collideFlat()

		HKNP_PADDING_BYTES(m_padding3, 1);

#if defined(HK_REAL_IS_DOUBLE)

		HKNP_PADDING_BYTES(m_padding4, 16);
#endif
};


/// A collision cache for convex convex interactions with face index information.
/// This is used when face information is not cached by the convex shapes.
struct hknpConvexConvexManifoldCollisionCacheWithFaceIndex : public hknpConvexConvexManifoldCollisionCache
{
	hkUint32	m_faceIdA;
	hkUint32	m_faceIdB;
	HKNP_PADDING_BYTES(m_padding5, sizeof(hkVector4) - 2 * sizeof(hkUint32) );
};


// ---------------------------------------------------------------------------------------------------------------------
// Composite caches
// ---------------------------------------------------------------------------------------------------------------------

/// Common base class for Convex-Composite and Composite-Composite
struct hknpCompositeCollisionCache : public hknpCollisionCache
{
	public:

		struct NmpBuffer
		{
			HK_ALIGN_REAL( hkUint8 m_buffer[32] );
		};

	public:

		void destructCdCacheImpl(
			const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
			hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,
			hknpCdBodyBase* HK_RESTRICT cdBodyA, hknpCdBodyBase* HK_RESTRICT cdBodyB,
			hknpCdCacheDestructReason::Enum reason );

		HK_FORCE_INLINE void resetCdCacheImpl();

		HK_FORCE_INLINE hknpQueryAabbNmp* getQueryAabbNmp() { return (hknpQueryAabbNmp*)&m_nmpBuffer; }

		HK_FORCE_INLINE void resetNmpState() { hknpQueryAabbNmpUtil::clearNmp(getQueryAabbNmp()); m_nmpTimeToLive = 0; m_numHits = 0; }

	public:

		hkUint32 m_numHits;			///< The number of child Convex-Convex collision caches.
		hkUint8 m_nmpTimeToLive;	// internal NMP counter
		HKNP_PADDING_BYTES(m_padding6, 11);
		hknpCdCacheRange m_childCdCacheRange;	///< Child cache range.
		NmpBuffer m_nmpBuffer;					///< NMP state.
};


/// Cache used for Convex-Composite collisions.
struct hknpConvexCompositeCollisionCache : hknpCompositeCollisionCache
{
	public:

		enum { NMP_TIME_TO_LIVE = 0x00ff };

	public:

		/// Construct a convex-composite collision cache in the given cache writer
		static void HK_CALL construct(
			const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB,
			hknpCdCacheWriter* cacheWriterInOut );

	public:

		/// Returns the size in bytes.
		HK_FORCE_INLINE int getSizeInBytes() const { return sizeof(*this); }

		/// Constructor.
		HK_FORCE_INLINE void init();

		/// Reset implementation.
		HK_FORCE_INLINE void resetCdCacheImpl();

	private:

		hknpConvexCompositeCollisionCache() {}
};


/// Cache used to collide a composite shape vs another composite shape
struct hknpCompositeCompositeCollisionCache : hknpCompositeCollisionCache
{
	public:

		/// Construct a composite-composite collision cache in the given cache writer
		static void HK_CALL construct(
			const hknpWorld& world, const hknpBody& bodyA, const hknpBody& bodyB,
			hknpCdCacheWriter* cacheWriterInOut );

	public:

		/// Returns the size in bytes.
		HK_FORCE_INLINE int getSizeInBytes() const { return sizeof(*this); }

		/// Constructor.
		HK_FORCE_INLINE void init();

		/// Reset implementation.
		HK_FORCE_INLINE void resetCdCacheImpl();

	private:

		hknpCompositeCompositeCollisionCache() {}
};


/// Structure within the childCdCacheStream referenced by hknpCompositeCompositeCollisionCache to start a new child
/// convex objectA
struct hknpSetShapeKeyACollisionCache : hknpCollisionCache
{
	public:

		struct NmpBuffer
		{
			HK_ALIGN_REAL( hkUint8 m_buffer[32] );
		};

	public:

		HK_FORCE_INLINE void init( hknpShapeKey shapeKeyA )
		{
			hknpCollisionCache::init( hknpCollisionCacheType::SET_SHAPE_KEY_A, sizeof(hknpSetShapeKeyACollisionCache) );
			setShapeKey( shapeKeyA );
			m_nmpTimeToLive = 0;
		}

		HK_FORCE_INLINE void resetCdCacheImpl() {}

		HK_FORCE_INLINE void getLeafShapeKeysImpl(
			HK_PAD_ON_SPU(hknpShapeKey)* shapeKeyA, HK_PAD_ON_SPU(hknpShapeKey)* shapeKeyB ) const
		{
			*shapeKeyA = getShapeKey();
		}

		HK_FORCE_INLINE hknpQueryAabbNmp* getQueryAabbNmp() { return (hknpQueryAabbNmp*)&m_nmpBuffer; }

	public:

		hkUint32    m_numHitsShapeKeyA;	///< Number of hits having this m_shapeKeyA
		hkUint8		m_nmpTimeToLive;
		HKNP_PADDING_BYTES(m_padding7, 1);

		hknpMaterialId m_materialIdA;
		hknpLinearTimType m_tim;
		HKNP_PADDING_BYTES(m_padding8, 2);

		hkUint32	m_collisionFilterInfo;
#if defined(HK_REAL_IS_DOUBLE)
		HKNP_PADDING_BYTES(m_padding9, 8);
#endif
		NmpBuffer	m_nmpBuffer;
};


#if defined(HK_PLATFORM_WIN32)
#	pragma warning( disable : 4820 )
#endif


/// Some constants.
enum
{
	/// The maximum theoretical size of a Convex-Convex cache.
	HKNP_MAX_CVX_CVX_CACHE_SIZE =
		sizeof(hknpConvexConvexManifoldCollisionCacheWithFaceIndex) +
		hknpConvexConvexCollisionCache::MAX_PROPERTY_BUFFER_SIZE,

	/// The maximum theoretical size of a hknpCollisionCache.
	HKNP_MAX_COLLISION_CACHE_SIZE = HKNP_MAX_CVX_CVX_CACHE_SIZE
};

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.inl>

#endif // HKNP_COLLISION_CACHE_H

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

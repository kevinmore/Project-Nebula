/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_H
#define HKNP_SHAPE_H

#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/Types/Properties/hkRefCountedProperties.h>
#include <Common/Base/Types/hkSignalSlots.h>
#include <Common/Base/Container/Array/hkFixedCapacityArray.h>
#include <Common/GeometryUtilities/Inertia/hkCompressedInertiaTensor.h>

#include <Physics/Physics/hknpTypes.h>
#include <Physics/Physics/Collide/Shape/hknpShapeType.h>
#include <Physics/Physics/Collide/Shape/VirtualTableUtil/hknpShapeVirtualTableUtil.h>

class hknpCollisionQueryCollector;
class hknpConvexShape;
class hknpConvexPolytopeShape;
class hknpCompositeShape;
class hknpHeightFieldShape;
class hknpQueryAabbNmp;
struct hkGeometry;
struct hknpAabbQuery;
struct hknpCollisionQueryContext;
struct hknpRayCastQuery;
struct hknpShapeQueryInfo;
struct hknpQueryFilterData;
struct hknpShapeCastQuery;
struct hknpShapeCollector;
struct hknpShapeKeyMask;
namespace hkcdGsk { struct Cache; }

#if defined( HK_PLATFORM_SPU )
	class hknpSpuSdfContactPointWriter;
#endif


/// An abstract iterator to enumerate shape keys in a composite shape.
class hknpShapeKeyIterator : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Destructor.
		virtual ~hknpShapeKeyIterator() {}

		/// Advance to the next shape key.
		virtual void next() = 0;

		/// Is the iterator still valid?
		/// While the iterator is still valid, next() can called to advance the shape key.
		HK_FORCE_INLINE bool isValid() const;

		/// Get the current shape key.
		HK_FORCE_INLINE hknpShapeKey getKey() const;

		/// Get the current shape key path.
		HK_FORCE_INLINE const hknpShapeKeyPath& getKeyPath() const;

	protected:

		/// Constructor.
		HK_FORCE_INLINE hknpShapeKeyIterator( const hknpShape& shape, const hknpShapeKeyMask* mask );

		/// The shape which created the iterator.
		const hknpShape* m_shape;

		/// A shape key mask defining which keys are enabled. This can be HK_NULL.
		const hknpShapeKeyMask* m_mask;

		/// The current shape key path.
		hknpShapeKeyPath m_keyPath;
};


/// The base class of all shapes.
class hknpShape : public hkReferencedObject
{
	//+version(1)

	public:

		/// Constants.
		enum
		{
			MAX_NUM_VERTICES_PER_FACE = 16,	// used by convex shapes
		};

		/// Internal flags describing the shape.
		enum FlagsEnum
		{
			IS_CONVEX_SHAPE					= 1<<0,	///< This shape can be cast to hknpConvexShape.
			IS_CONVEX_POLYTOPE_SHAPE		= 1<<1,	///< This shape can be cast to hknpConvexPolytopeShape.
			IS_COMPOSITE_SHAPE				= 1<<2,	///< This shape can be cast to hknpCompositeShape.
			IS_HEIGHT_FIELD_SHAPE			= 1<<3,	///< This shape can be cast to hknpHeightFieldShape.
			USE_SINGLE_POINT_MANIFOLD		= 1<<4,	///< Force any manifold using this shape to have a single contact point.
			IS_TRIANGLE_OR_QUAD_NO_EDGES	= 1<<5,	///< The shape is a triangle/quad shape and we don't need to collide the edges.
			SUPPORTS_BPLANE_COLLISIONS		= 1<<6,	///< The shape supports fast, low quality collisions with concave triangles.
			USE_NORMAL_TO_FIND_SUPPORT_PLANE= 1<<7, ///< The shape supports using the normal to find support plane.
			USE_SMALL_FACE_INDICES			= 1<<8, ///< The shape returns 8-bit face indices from getSupportingFace().
			NO_GET_ALL_SHAPE_KEYS_ON_SPU	= 1<<9, ///< The shape's getAllShapeKeys() method has *NOT* been implemented on SPU.
			SHAPE_NOT_SUPPORTED_ON_SPU		= 1<<10,///< The shape's implementation is not available on SPU.
		};
		typedef hkFlags< FlagsEnum, hkUint16 > Flags;

		/// Flags describing how a mutable shape was mutated.
		enum MutationFlagsEnum
		{
			MUTATION_AABB_CHANGED			  = 1<<0,	///< The AABB of the shape has changed.
			MUTATION_DISCARD_CACHED_DISTANCES = 1<<1,	///< The shape has changed such that cached distances must be discarded.
			MUTATION_REBUILD_COLLISION_CACHES = 1<<2,	///< Rebuild any collision caches using the shape.
		};
		typedef hkFlags< FlagsEnum, hkUint8 > MutationFlags;

		/// Modes for scaling some shapes.
		enum ScaleMode
		{
			SCALE_SURFACE,		///< Scale the collision surface, considering both the vertices and the convex radius.
			SCALE_VERTICES		///< Scale the vertices only, ignoring the convex radius value. This is the faster mode.
		};

		/// Modes for representing convex radius expansion in shape display geometry.
		enum ConvexRadiusDisplayMode
		{
			CONVEX_RADIUS_DISPLAY_NONE,		///< Generate simple geometry using the shape vertices, as if the radius were zero.
			CONVEX_RADIUS_DISPLAY_PLANAR,	///< Expand the faces by the radius, producing inaccurate sharp corners but no extra faces.
			CONVEX_RADIUS_DISPLAY_ROUNDED	///< Expand the vertices by the radius, producing accurate round corners but at the cost of extra faces.
		};

		/// A query structure for colliding spheres with a signed distance field. See getSignedDistances().
		struct SdfQuery
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, SdfQuery );

			const hkVector4* m_sphereCenters;	///< An array of sphere centers to query against the SDF.
			int				 m_numSpheres;		///< The number of query spheres.
			hkReal			 m_spheresRadius;	///< The radius of all the query spheres.
			hkReal			 m_maxDistance;		///< Spheres which produce a bigger distance than this can be ignored, they only should set the distance to HK_REAL_MAX.
		};

		/// A contact point returned by a signed distance field query.
		struct SdfContactPoint
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, SdfContactPoint );

			hkVector4	  m_position;	///< The contact point on the SDF shape.
			hkVector4     m_normal;		///< The collision normal pointing away from the SDF shape.
			hkReal		  m_distance;	///< The distance between the query sphere and the SDF shape.
			hknpVertexId  m_vertexId;	///< The query sphere index.

			hknpShapeKey  m_shapeKey;	///< The shape key output identifying the hit region. This only needs to be returned if the SDF implements getLeafShape().
			hknpShapeTag  m_shapeTag;	///< The shape tag of the hit. Used for looking up the material.
		};

		/// Configuration for buildMassProperties().
		struct MassConfig
		{
			/// A quality level used by mass properties calculations.
			enum Quality
			{
				QUALITY_LOW,	///< Estimate the mass properties using the AABB of the root shape.
				QUALITY_MEDIUM,	///< Estimate the mass properties using the AABBs of all enabled leaf shapes.
				QUALITY_HIGH	///< Calculate the mass properties from the volume of all enabled leaf shapes.
			};

			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, MassConfig );

			/// Create a mass config with a desired density value.
			static HK_FORCE_INLINE MassConfig fromDensity( hkReal density, hkReal inertiaFactor=1.5f, Quality quality=QUALITY_HIGH );

			/// Create a mass config with a desired mass value.
			static HK_FORCE_INLINE MassConfig fromMass( hkReal mass, hkReal inertiaFactor=1.5f, Quality quality=QUALITY_HIGH );

			/// Constructor. Sets default values, with a desired density of 1.0.
			HK_FORCE_INLINE MassConfig();

			/// Calculate a mass from the given volume.
			HK_FORCE_INLINE hkReal calcMassFromVolume( hkReal volume ) const;

			/// Controls the quality of the mass property calculations (default = QUALITY_HIGH).
			Quality m_quality;

			/// Scales the inertia without scaling the mass.
			/// Values > 1.0 distribute the mass towards the surface of the shape, emulating the inertia of a "hollow"
			/// shape, and increasing solver stability (default = 1.5).
			hkReal m_inertiaFactor;

		protected:

			HK_FORCE_INLINE MassConfig( hkReal massOrNegativeDensity, hkReal inertiaFactor, Quality quality );

			hkReal m_massOrNegativeDensity;
		};

		/// Configuration for buildSurfaceGeometry().
		struct BuildSurfaceGeometryConfig
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, BuildSurfaceGeometryConfig );

			BuildSurfaceGeometryConfig() { m_radiusMode = CONVEX_RADIUS_DISPLAY_ROUNDED; m_storeShapeKeyInTriangleMaterial = false; }
			BuildSurfaceGeometryConfig( ConvexRadiusDisplayMode rm ) { m_radiusMode = rm; m_storeShapeKeyInTriangleMaterial = false; }

			/// An indication about how the convex radius should be displayed if the shape supports it.
			ConvexRadiusDisplayMode m_radiusMode;	

			/// By default the material returned with each generated triangle is the shape tag.
			/// If this parameter is set to true, the triangle material will store the shape key instead.
			hkBool m_storeShapeKeyInTriangleMaterial;
		};

		/// A set of signals fired by mutable shapes.
		struct MutationSignals
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, MutationSignals );

			HK_DECLARE_SIGNAL( ShapeMutatedSignal, hkSignal1<hkUint8> );	// See MutationFlags.
			HK_DECLARE_SIGNAL( ShapeDestroyedSignal, hkSignal0 );

			ShapeMutatedSignal		m_shapeMutated;		///< Fired just after the shape is mutated.
			ShapeDestroyedSignal	m_shapeDestroyed;	///< Fired just before the shape is destroyed.
		};

#if !defined(HK_PLATFORM_SPU)
		typedef hkBlockStream<SdfContactPoint>::Writer SdfContactPointWriter;
#else
		typedef hknpSpuSdfContactPointWriter SdfContactPointWriter;
#endif

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

#if !defined(HK_PLATFORM_SPU)

		/// Constructor.
		HK_FORCE_INLINE hknpShape( hknpCollisionDispatchType::Enum dispatchType );

		/// Serialization constructor.
		hknpShape( hkFinishLoadedObjectFlag flag );

		/// Destructor.
		virtual ~hknpShape();

#endif

		/// Get the concrete shape type.
		/// NOTE: m_dispatchType specifies how the shape is used during collision detection.
		virtual hknpShapeType::Enum getType() const;

		/// Compute an AABB for the shape in \a transform space.
		virtual void calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const;

		/// Calculate the shape size in bytes. Used to transfer shapes to SPU.
		virtual int calcSize() const;

#if defined( HK_PLATFORM_PPU )

		/// Automatically set the SPU flags on this shape.
		virtual void computeSpuFlags();

#endif

#if !defined(HK_PLATFORM_SPU)

		/// Mutable shapes must return a non-null pointer to a MutationSignals structure.
		virtual MutationSignals* getMutationSignals();

#endif


		// -------------------------------------------------------------------------------------------------------------
		// Convex interface
		// -------------------------------------------------------------------------------------------------------------

		/// Get the number of vertices written out by getSupportVertices().
		virtual int getNumberOfSupportVertices() const;

		/// Get a pointer to the first support vertex.
		/// Implementations of this method *may* use the \a vertexBuffer, but need not.
		/// You therefore cannot rely on \a vertexBuffer getting filled with any meaningful data!
		virtual const hkcdVertex* getSupportVertices( hkcdVertex* vertexBuffer, int bufferSize ) const;

		/// Get the supporting vertex in a given direction.
		virtual void getSupportingVertex( hkVector4Parameter direction, hkcdVertex* vertexOut ) const;

		/// Extract the vertices for a given list of vertex IDs.
		virtual void convertVertexIdsToVertices( const hkUint8* ids, int numIds, hkcdVertex* verticesOut ) const;

		/// Get the number of faces.
		/// This should return zero if the shape does not store faces.
		virtual int getNumberOfFaces() const;

		/// Get the vertices of a single face. Returns the number of vertices.
		virtual int getFaceVertices( const int faceIndex, hkVector4& planeOut, hkcdVertex* vertexBufferOut ) const;

		/// Get the plane equation of a single face.
		virtual void getFaceInfo( const int faceIndex, hkVector4& planeOut, int& minAngleOut ) const;

		/// Get the face index from a supporting surface point.
		virtual int getSupportingFace(
			hkVector4Parameter surfacePoint, const hkcdGsk::Cache* gskCache, bool useB,
			hkVector4& planeOut, int& minAngleOut, hkUint32& prevFaceId ) const;

		/// Calculate the minimum angle between any pair of adjacent faces in the shape.
		virtual hkReal calcMinAngleBetweenFaces() const;


		// -------------------------------------------------------------------------------------------------------------
		// Signed distance field interface
		// -------------------------------------------------------------------------------------------------------------

		/// Get the signed distances from an array of input spheres.
		virtual void getSignedDistances( const hknpShape::SdfQuery& input, SdfContactPoint* contactsOut ) const;

		/// Get the signed distances from the support vertices of a query shape, and write them to a contact stream.
		///	Return the total number of support vertices (not the number of contacts written) - e.g. a convex object
		/// should return it's number of vertices.
		/// Notes:
		///    - A query sphere is allowed to return multiple hits.
		///    - If the distance of a query point is bigger than maxDistance, no hit will be reported.
		///    - The output vertexIds must be sorted.
		virtual int getSignedDistanceContacts(
			const hknpSimulationThreadContext& tl, const hknpShape* queryShape, const hkTransform& sdfFromQueryTransform,
			hkReal maxDistance, int vertexIdOffset, SdfContactPointWriter& contactPointsOut ) const;

		HK_AUTO_INLINE int getSignedDistanceContactsImpl(
			const hknpSimulationThreadContext& tl, const hknpShape* queryShape, const hkTransform& sdfFromQueryTransform,
			hkReal maxDistance, int vertexIdOffset, SdfContactPointWriter& contactPointsOut ) const;


		// -------------------------------------------------------------------------------------------------------------
		// Composite interface
		// -------------------------------------------------------------------------------------------------------------

#if !defined(HK_PLATFORM_SPU)

		/// Create a shape key mask for use with getAllShapeKeys() and createShapeKeyIterator().
		/// If the shape does not support masking, this should return HK_NULL.
		virtual hknpShapeKeyMask* createShapeKeyMask() const;

#else

		virtual void patchShapeKeyMaskVTable( hknpShapeKeyMask* mask ) const;

#endif

		/// Get all shape keys.
		/// If a mask is provided, this should skip any disabled keys.
#if !defined(HK_PLATFORM_SPU)
		virtual void getAllShapeKeys(
			const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,
			hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const;	
#else
		virtual void getAllShapeKeys(
			const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,
			hkUint8* shapeBuffer, int shapeBufferSize,
			hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const;	
#endif

		/// Create a shape key iterator to enumerate all shape keys.
		/// If a mask is provided, the iterator should skip any disabled keys.
#if !defined( HK_PLATFORM_SPU )
		virtual hkRefNew<hknpShapeKeyIterator> createShapeKeyIterator(
			const hknpShapeKeyMask* mask = HK_NULL ) const;
#else
		virtual hknpShapeKeyIterator* createShapeKeyIterator(
			hkUint8* buffer, int bufferSize, const hknpShapeKeyMask* mask = HK_NULL ) const;
#endif

		/// Get the leaf shape identified by a shape key.
		/// The collector stores the resulting shape pointer as well as storage for shapes that are created on demand.
		/// Notes:
		///  - If this shape is a convex shape, the collector's shape pointer will point to THIS.
		///  - m_shapeTagOut as returned in hknpShapeCollector is only the leaf shape's local tag. To decode the global
		///    tag data for the shape you need to call hknpShapeTagCodec::decode() using the collector's m_shapeTagPath.
		virtual void getLeafShape( hknpShapeKey key, hknpShapeCollector* collector ) const;


		// -------------------------------------------------------------------------------------------------------------
		// Query interface
		// -------------------------------------------------------------------------------------------------------------

		/// This is an internal method.
		/// Please use hknpShapeQueryInterface::castRay() instead and see there for details.
		virtual void castRayImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpRayCastQuery& query,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector ) const;

		/// This is an internal method.
		/// Please use hknpShapeQueryInterface::queryAabb() instead and see there for details.
		virtual void queryAabbImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hkArray<hknpShapeKey>* hits, hknpQueryAabbNmp* nmpInOut ) const;

		/// This is an internal method.
		/// Please use hknpShapeQueryInterface::queryAabb() instead and see there for details.
		virtual void queryAabbImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector, hknpQueryAabbNmp* nmpInOut ) const;


		// -------------------------------------------------------------------------------------------------------------
		// Properties
		// -------------------------------------------------------------------------------------------------------------

		/// Attach a property to the shape.
		/// Replaces any existing property using the same key.
		HK_FORCE_INLINE void setProperty( hknpPropertyKey propertyKey, hkReferencedObject* propertyObject );

		/// Get a property from the shape.
		/// Returns HK_NULL if there is no property attached with the given key.
		HK_FORCE_INLINE hkReferencedObject* getProperty( hknpPropertyKey propertyKey ) const;

		/// Build and attach mass properties to the shape.
		/// Replaces any existing mass properties.
		HK_FORCE_INLINE void setMassProperties( const MassConfig& massConfig );

		/// Attach mass properties to the shape.
		/// Replaces any existing mass properties.
		HK_FORCE_INLINE void setMassProperties( const hkDiagonalizedMassProperties& massProperties );

		/// Get the mass properties of the shape, if any.
		/// Returns HK_FAILURE if no hknpShapeMassProperties property is attached.
		HK_FORCE_INLINE hkResult getMassProperties( hkDiagonalizedMassProperties& massPropertiesOut ) const;
		HK_FORCE_INLINE hkResult getMassProperties( hkMassProperties& massPropertiesOut ) const;


		// -------------------------------------------------------------------------------------------------------------
		// Utilities
		// -------------------------------------------------------------------------------------------------------------

#if !defined(HK_PLATFORM_SPU)

		/// Build mass properties for the shape, based on the given configuration.
		/// The default implementation approximates the shape volume using its AABB.
		virtual void buildMassProperties(
			const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const;

		/// Build a geometry representing the collision surface of the shape. Returns HK_FAILURE if not supported.
		/// This method is only used to visualize shapes, it is not used by the simulation.
		virtual hkResult buildSurfaceGeometry(
			const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const = 0;

		/// Check consistency/validity of the shape.
		/// Fires an assert if any inconsistency is found.
		virtual void checkConsistency() const;

		/// Check whether the shape is mutable (has mutation signals).
		HK_FORCE_INLINE bool isMutable() const;

#endif

		/// Get the internal flags for this shape.
		HK_FORCE_INLINE Flags getFlags() const;

		/// Set the internal flags. Handle with care.
		HK_FORCE_INLINE void setFlags( Flags flags );

		/// Get the value of m_numShapeKeyBits.
		HK_FORCE_INLINE hkUint8 getNumShapeKeyBits() const;

		/// Cast to a hknpConvexShape if possible.
		HK_FORCE_INLINE const hknpConvexShape* asConvexShape() const;

		/// Cast to a hknpConvexPolytopeShape if possible.
		HK_FORCE_INLINE const hknpConvexPolytopeShape* asConvexPolytopeShape() const;

		/// Cast to a hknpCompositeShape if possible.
		HK_FORCE_INLINE const hknpCompositeShape* asCompositeShape() const;

		/// Cast to a hknpHeightFieldShape if possible.
		HK_FORCE_INLINE const hknpHeightFieldShape* asHeightFieldShape() const;

	protected:

#if defined(HK_PLATFORM_HAS_SPU)

		/// Empty constructor, required for the virtual table utility on PlayStation(R)3.
		hknpShape( hknpShapeVirtualTableUtilDummy dummy ) {}

#endif

		/// Internal method used to initialize a shape without constructing it. Required for SPU.
		HK_FORCE_INLINE void init( hknpCollisionDispatchType::Enum dispatchType );


		// -------------------------------------------------------------------------------------------------------------
		// Member variables
		// -------------------------------------------------------------------------------------------------------------

	protected:

		/// Internal flags.
		HK_ALIGN16( Flags ) m_flags;

		/// The maximum number of bits required to store any shape key local to this shape.
		/// This does NOT include bits used by child shapes (if any).
		/// To mask out a key for this shape, use the following formulae:
		///  - mask = ((1 << m_numShapeKeyBits) - 1) << (32 - m_numShapeKeyBits)
		///  - masked key = (key << (num shape key bits of parents if any)) & mask
		hkUint8 m_numShapeKeyBits;

	public:

		/// A type specifying how this shape is used during collision detection.
		/// See \a hknpCollisionDispatchType.
		hkEnum< hknpCollisionDispatchType::Enum, hkUint8 > m_dispatchType;

		/// A radius by which to expand convex shape vertices.
		/// If the shape is a sphere or a capsule, this is the actual radius.
		hkReal m_convexRadius;

		/// User data. Not used by the engine.
		/// Defaults to zero.
		mutable hkUint64 m_userData;

	protected:

		/// Optional properties attached to the shape.
		/// Defaults to HK_NULL.
		hkRefCountedProperties* m_properties;
};


/// An interface for enabling and disabling shape keys.
struct hknpShapeKeyMask
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeKeyMask );

#if defined(HK_PLATFORM_SPU)

	hknpShapeKeyMask() {}

	/// Empty constructor, required for the virtual table utility on PlayStation(R)3.
	hknpShapeKeyMask( hknpShapeKeyMaskVirtualTableDummy dummy ) {}

#else

	/// Destructor.
	virtual ~hknpShapeKeyMask() {}

	/// Enable or disable a shape key.
	virtual	void setShapeKeyEnabled( hknpShapeKey key, bool isEnabled ) = 0;

	/// Commit changes.
	/// Note: this should return true if ANY of the keys is enabled.
	virtual bool commitChanges() = 0;

#endif

	/// Returns a shape key state, enabled (true) or disabled (false).
	virtual bool isShapeKeyEnabled( hknpShapeKey key ) const = 0;

	/// Returns the size. Mainly required for PlayStation(R)3.
	virtual int calcSize() const = 0;
};


/// A reference counted mass properties object that may be attached as a property to shapes.
struct hknpShapeMassProperties : public hkReferencedObject
{
	HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
	HK_DECLARE_REFLECTION();

	hknpShapeMassProperties() {}
	hknpShapeMassProperties( hkFinishLoadedObjectFlag flag );

	virtual ~hknpShapeMassProperties() {}

	/// The mass properties
	HK_ALIGN( hkCompressedMassProperties m_compressedMassProperties, 8 );
};


#if defined( HK_PLATFORM_SPU )
#	include <Physics/Physics/Collide/NarrowPhase/Detector/SignedDistanceField/hknpSpuSdfContactPointWriter.h>
#endif


#include <Physics/Physics/Collide/Shape/hknpShape.inl>


#endif // HKNP_SHAPE_H

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

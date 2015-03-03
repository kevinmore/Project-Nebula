/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SCALED_CONVEX_SHAPE_H
#define HKNP_SCALED_CONVEX_SHAPE_H

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>

extern const hkClass hknpScaledConvexShapeClass;


/// Scaled convex shape base class.
/// For internal use only.
class hknpScaledConvexShapeBase : public hknpShape
{
	//+version(1)

	public:

		/// Create a scaled convex shape in the given buffer. \a buffer must be 16 byte aligned.
		/// Advanced use only.
		static HK_FORCE_INLINE hknpScaledConvexShapeBase* createInPlace(
			const hknpConvexShape* childShape, hkVector4Parameter scale, ScaleMode mode, hkUint8* buffer, int bufferSize );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

#if !defined(HK_PLATFORM_SPU)

		/// Constructor.
		hknpScaledConvexShapeBase(
			const hknpConvexShape* childShape, hkVector4Parameter scale, ScaleMode mode = SCALE_SURFACE );

		/// Serialization constructor.
		hknpScaledConvexShapeBase( class hkFinishLoadedObjectFlag flag );

#endif

		/// Set the scale.
		void HK_FORCE_INLINE setScale( hkVector4Parameter scale, ScaleMode mode );

		/// Get the scale.
		HK_FORCE_INLINE const hkVector4& getScale() const;

		/// Get the translational offset.
		HK_FORCE_INLINE const hkVector4& getTranslation() const;

		/// Returns the unscaled convex shape.
		HK_FORCE_INLINE const hknpConvexShape* getChildShape() const;

		/// Utility method to scale a single vertex
		HK_FORCE_INLINE void scaleVertex( hkVector4Parameter vertex, hkVector4& scaledVertexOut );

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::SCALED_CONVEX; }

		virtual void calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const HK_OVERRIDE;
		virtual int calcSize() const HK_OVERRIDE;

		virtual int					getNumberOfSupportVertices() const HK_OVERRIDE;
		virtual const hkcdVertex*   getSupportVertices( hkcdVertex* vertexBuffer, int bufferSize ) const HK_OVERRIDE;
		virtual void				getSupportingVertex( hkVector4Parameter direction, hkcdVertex* vertexOut ) const HK_OVERRIDE;
		virtual void				convertVertexIdsToVertices( const hkUint8* ids, int numVerts, hkcdVertex* verticesOut ) const HK_OVERRIDE;
		virtual int					getNumberOfFaces() const HK_OVERRIDE;
		virtual int					getFaceVertices( const int faceIndex, hkVector4& planeOut, hkcdVertex* vertexBufferOut ) const HK_OVERRIDE;
		virtual void				getFaceInfo( const int index, hkVector4& planeOut, int& minAngleOut ) const HK_OVERRIDE;
		virtual int					getSupportingFace(
			hkVector4Parameter surfacePoint, const hkcdGsk::Cache* gskCache, bool useB,
			hkVector4& planeOut, int& minAngleOut, hkUint32 &prevFaceId ) const HK_OVERRIDE;
		virtual hkReal				calcMinAngleBetweenFaces() const HK_OVERRIDE;

#if !defined(HK_PLATFORM_SPU)

		virtual hknpShapeKeyMask* createShapeKeyMask() const HK_OVERRIDE { return HK_NULL; }

		virtual hkRefNew<hknpShapeKeyIterator> createShapeKeyIterator( const hknpShapeKeyMask* mask ) const HK_OVERRIDE;

#endif

		virtual void getLeafShape( hknpShapeKey key, hknpShapeCollector* collector ) const HK_OVERRIDE;

		virtual void queryAabbImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hkArray<hknpShapeKey>* hits, hknpQueryAabbNmp* nmpInOut ) const HK_OVERRIDE;

		virtual void queryAabbImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector, hknpQueryAabbNmp* nmpInOut ) const HK_OVERRIDE;

		virtual void castRayImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpRayCastQuery& query,
			const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector ) const HK_OVERRIDE;

#if !defined(HK_PLATFORM_SPU)

		virtual void buildMassProperties(
			const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const HK_OVERRIDE;

		virtual hkResult buildSurfaceGeometry(
			const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const HK_OVERRIDE;

#endif

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpScaledConvexShapeBase, hknpShape );

		
		HK_FORCE_INLINE hkReal boundMinAngle( const hkReal angle ) const;

		/// Used to initialize a scaled convex shape without constructing it. Required for SPU
		HK_FORCE_INLINE void init( const hknpConvexShape* childShape, hkVector4Parameter scale, ScaleMode mode );

	protected:

		/// The wrapped convex shape.
		HK_ON_SPU( mutable ) const hknpConvexShape* m_childShape;

		/// A computed scale value. May differ from the user provided scale.
		hkVector4 m_scale;

		/// A computed internal offset added to vertices after multiplying by the scale to adjust convex radius scaling.
		hkVector4 m_translation;

		/// The size of the convex child shape. This value is only valid and used on PlayStation(R)3. On SPU this value
		/// will be reset to 0 once the child shape has been uploaded to SPU.
		mutable int m_childShapeSize;
};


/// Scaled convex shape.
/// This shape wraps a hknpConvexShape with a non-uniform scale. All convex shape methods are passed
/// onto the wrapped shape, with their inputs and/or outputs transformed appropriately.
class hknpScaledConvexShape : public hknpScaledConvexShapeBase
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

	#if !defined(HK_PLATFORM_SPU)

		/// Constructor.
		hknpScaledConvexShape(
			const hknpConvexShape* childShape, hkVector4Parameter scale, ScaleMode mode = SCALE_SURFACE );

		/// Serialization constructor.
		hknpScaledConvexShape( class hkFinishLoadedObjectFlag flag );

	#endif

		/// Destructor.
		~hknpScaledConvexShape();

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpScaledConvexShape, hknpScaledConvexShapeBase );
};

#include <Physics/Physics/Collide/Shape/Convex/Scaled/hknpScaledConvexShape.inl>

#endif // HKNP_SCALED_CONVEX_SHAPE_H

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

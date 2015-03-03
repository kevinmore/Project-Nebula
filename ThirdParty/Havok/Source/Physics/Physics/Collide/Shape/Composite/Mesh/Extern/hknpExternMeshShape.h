/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_EXTERN_MESH_SHAPE_H
#define HKNP_EXTERN_MESH_SHAPE_H

#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>

// Opaque bounding volume tree.
struct hknpExternMeshShapeTree;

extern const class hkClass hknpExternMeshShapeClass;
extern const class hkClass hknpExternMeshShapeTreeClass;


/// A mesh shape referencing an externally stored mesh of arbitrary format.
class hknpExternMeshShape : public hknpCompositeShape
{
	public:

		/// Mesh data interface.
		/// You must implement this class in order to give the shape access to your external data.
		class Mesh : public hkReferencedObject
		{
			public:

				HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
				HK_DECLARE_REFLECTION();

				/// Constructor.
				Mesh();

				/// Serialization constructor.
				Mesh( hkFinishLoadedObjectFlag flag );

				/// Destructor.
				virtual	~Mesh();

				/// This method should return the total number of triangles in the external data.
				virtual int getNumTriangles() const = 0;

				/// This method should fill the \a verticesOut array with the 3 vertices of the triangle
				/// referenced by \a index. Use hknpExternMeshShape::getTriangleIndex() to obtain the triangle index
				/// from a shape key.
				virtual void getTriangleVertices( int index, hkVector4* verticesOut ) const = 0;

				/// This method should return the shape tag for the triangle referenced by \a index.
				virtual hknpShapeTag getTriangleShapeTag( int index ) const = 0;
		};

	public:

		/// Build a bounding volume tree for the given source geometry.
		static hknpExternMeshShapeTree* buildTree( const Mesh* source );

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Constructor.
		/// A precreated bounding volume tree may be provided, otherwise one will be built.
		hknpExternMeshShape( const Mesh* mesh, const hknpExternMeshShapeTree* tree = HK_NULL );

		/// Serialization constructor.
		hknpExternMeshShape( hkFinishLoadedObjectFlag flag );

		/// Destructor.
		~hknpExternMeshShape();

		/// Get the mesh data interface. Not implemented on SPU, you need to implement it yourself to ensure SPU access.
		HK_ON_CPU(HK_FORCE_INLINE) const Mesh* getMesh() const;

		/// Get the bounding tree. Use this method to ensure SPU access.
		HK_ON_CPU(HK_FORCE_INLINE) const hknpExternMeshShapeTree* getTree() const;

		/// Get the triangle index corresponding to a shape key.
		HK_FORCE_INLINE int getTriangleIndex( hknpShapeKey key ) const;

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE;

		virtual void calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const HK_OVERRIDE;

		virtual int calcSize() const HK_OVERRIDE;

#if !defined(HK_PLATFORM_SPU)

		virtual hkRefNew<hknpShapeKeyIterator> createShapeKeyIterator(
			const hknpShapeKeyMask* mask = HK_NULL ) const HK_OVERRIDE;

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

		virtual hkResult buildSurfaceGeometry(
			const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const HK_OVERRIDE;

#endif

		/// Internal. Cast a shape against a hknpExternMeshShape.
		static void castShapeImpl(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector );

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpExternMeshShape, hknpCompositeShape );

	protected:

		/// The mesh data interface. Aligned for SPU access.
		HK_ALIGN16(const Mesh* m_mesh);

		/// The bounding volume tree.
		const hknpExternMeshShapeTree* m_tree;

		/// Number of bits required to store tree leaf node indices, or 0 if disabled.
		hkUint8	m_numIndexKeyBits;

		/// True if the bounding volume tree was created by this shape.
		hkBool m_ownTree;

		friend struct hknpExternMeshShapeInternals;
};

#include <Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpExternMeshShape.inl>


#endif // HKNP_EXTERN_MESH_SHAPE_H

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

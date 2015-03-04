/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_TRIANGLE_SHAPE_H
#define HKNP_TRIANGLE_SHAPE_H

#include <Common/Base/Types/hkBaseTypes.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

#include <Physics/Physics/Collide/Shape/Convex/Polytope/hknpConvexPolytopeShape.h>

extern const hkClass hknpTriangleShapeClass;

// Disable warning about no public constructors in hknpTriangleShape
#if defined(HK_PLATFORM_SIM_SPU)
#	pragma warning( push )
#	pragma warning( disable : 4610 )
#endif


/// Triangle shape.
/// Note: triangles are normally created on-the-fly from mesh shapes, rather than being created by the user.
/// Note: hknpTriangleShape is actually a quad shape, with space for 4 vertices.
class hknpTriangleShape : public hknpConvexPolytopeShape
{
	public:

		/// Creates a triangle shape with uninitialized vertices.
		/// You can turn it into a proper triangle by calling setVertices().
		static hknpTriangleShape* HK_CALL createEmptyTriangleShape( hkReal radius = HKNP_SHAPE_DEFAULT_CONVEX_RADIUS );

#if !defined(HK_PLATFORM_SPU)

		/// Creates a triangle shape from 3 vertices.
		static HK_FORCE_INLINE hknpTriangleShape* HK_CALL createTriangleShape(
			hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c,
			hkReal radius = HKNP_SHAPE_DEFAULT_CONVEX_RADIUS );

#else

		/// Set the reference triangle shape. SPU only.
		static void HK_CALL setReferenceTriangleShape( hknpTriangleShape* pReferenceTriangleShape )
		{
			s_referenceTriangleShape = pReferenceTriangleShape;
		}

#endif

		/// Get the reference triangle shape.
		/// This is a triangle shape constructed once in a static buffer, used to quickly construct new triangle shapes.
		static HK_FORCE_INLINE const hknpTriangleShape* HK_CALL getReferenceTriangleShape();

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Serialization constructor
		hknpTriangleShape( hkFinishLoadedObjectFlag flag );

		/// Initializes a triangle shape.
		void setVertices( hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c );

		/// Initializes a quad shape.
		void setVertices( hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c, hkVector4Parameter d );

		/// Inlined version of setVertices() for triangle shape.
		HK_FORCE_INLINE void _setVertices(
			hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c );

		/// Inline version of setVertices() for quad shape.
		HK_FORCE_INLINE void _setVertices(
			hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c, hkVector4Parameter d );

		/// Converts this to a triangle using the first triangle of the quad.
		/// This is necessary if you initially called setVertices( hkVector4 v[4] ) but afterwards you decided that
		/// you actually wanted the setVertices( hkVector4 v[3] ) version.
		HK_FORCE_INLINE void _convertToSingleTriangle();

		/// Returns TRUE if this Triangle shape is actually a quad.
		HK_FORCE_INLINE bool isQuad() const;

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::TRIANGLE; }

		virtual int calcSize() const HK_OVERRIDE;

#if !defined(HK_PLATFORM_SPU)

		virtual void buildMassProperties(
			const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const;

#endif

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpTriangleShape, hknpConvexPolytopeShape );

#if !defined(HK_PLATFORM_SPU)

		/// Protected constructor. Use static creation methods instead.
		HK_FORCE_INLINE hknpTriangleShape( hkReal radius );

#endif

		friend class hknpInplaceTriangleShape;

#if defined(HK_PLATFORM_SPU)

		static hknpTriangleShape* s_referenceTriangleShape;

#endif
};


/// A hknpTriangleShape stored in a local buffer.
/// Allows for a triangle shape to be constructed on the stack.
class hknpInplaceTriangleShape
{
	public:

		enum
		{
			UNALIGNED_BUFFER_SIZE = sizeof(hknpTriangleShape)						+
									4 * sizeof(hkcdVertex)							+
									HK_NEXT_MULTIPLE_OF(4, 1) * sizeof(hkVector4)	+
									4 * sizeof(hknpConvexPolytopeShape::Face)		+
									1 * 4 * sizeof(hknpConvexPolytopeShape::VertexIndex),

			/// The buffer size required to hold a hknpTriangleShape, including its vertices, planes and faces.
			BUFFER_SIZE = HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, UNALIGNED_BUFFER_SIZE ),
		};

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpInplaceTriangleShape );

		/// Constructor.
		/// Initializes the local buffer by copying the reference triangle to it.
		HK_FORCE_INLINE hknpInplaceTriangleShape( hkReal radius = HKNP_SHAPE_DEFAULT_CONVEX_RADIUS );

		/// Get a pointer to the internal \a m_buffer that holds the hknpTriangleShape.
		HK_FORCE_INLINE hknpTriangleShape* getTriangleShape();

	private:

		// Private constructor.
		// Initializes the local buffer by constructing a triangle shape in place.
		hknpInplaceTriangleShape( bool dummy );

	protected:

		/// Buffer that holds a hknpTriangleShape.
		HK_ALIGN_REAL( hkUint8 m_buffer[BUFFER_SIZE] );

		friend class hknpTriangleShape;
};


// Restore previous state for warning about no public constructors in hknpTriangleShape
#if defined(HK_PLATFORM_SIM_SPU)
#	pragma warning( pop )
#endif

#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.inl>


#endif // HKNP_TRIANGLE_SHAPE_H

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

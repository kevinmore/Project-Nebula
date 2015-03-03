/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_MASKED_COMPOSITE_SHAPE_H
#define HKNP_MASKED_COMPOSITE_SHAPE_H

#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>


/// A shape which wraps another composite shape and adds the ability to enable and disable shape keys
/// of the wrapped shape on the fly, without modifying the wrapped shape directly.

class hknpMaskedCompositeShape : public hknpCompositeShape
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

		/// Constructor.
		hknpMaskedCompositeShape( const hknpCompositeShape* shape );

		/// Destructor.
		~hknpMaskedCompositeShape();

		/// Read-only access to the shape key mask.
		const hknpShapeKeyMask*	getMask() const { return m_mask; }

		/// Read-write access to the shape key mask.
		hknpShapeKeyMask* accessMask() { return &m_maskWrapper; }

		/// Returns the masked child shape.
		HK_AUTO_INLINE const hknpCompositeShape* getChildShape() const;

		//
		// hknpShape implementation
		//

		virtual hknpShapeType::Enum getType() const HK_OVERRIDE { return hknpShapeType::MASKED_COMPOSITE; }

		virtual void calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const HK_OVERRIDE;

		virtual int calcSize() const HK_OVERRIDE;

#if defined(HK_PLATFORM_PPU)

		/// Automatically set the SPU flags on this shape.
		virtual void computeSpuFlags() HK_OVERRIDE;

#endif

#if !defined(HK_PLATFORM_SPU)
		virtual hkRefNew<hknpShapeKeyIterator> createShapeKeyIterator(
			const hknpShapeKeyMask* mask = HK_NULL ) const HK_OVERRIDE;
#else
		hknpShapeKeyIterator* createShapeKeyIterator(
			hkUint8* buffer, int bufferSize,
			const hknpShapeKeyMask* mask = HK_NULL ) const HK_OVERRIDE;
#endif

#if !defined(HK_PLATFORM_SPU)
		virtual void getAllShapeKeys(
			const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,
			hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const HK_OVERRIDE;
#else
		virtual void getAllShapeKeys(
			const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,
			hkUint8* shapeBuffer, int shapeBufferSize,
			hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const HK_OVERRIDE;
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

		virtual MutationSignals* getMutationSignals() HK_OVERRIDE;

		virtual hkResult buildSurfaceGeometry( const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const HK_OVERRIDE;

#endif

	protected:

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpMaskedCompositeShape, hknpCompositeShape );

	public:

		/// A wrapper to intercept changes to an underlying shape key mask.
		struct MaskWrapper : public hknpShapeKeyMask
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, MaskWrapper );

#if !defined( HK_PLATFORM_SPU )
			virtual	void setShapeKeyEnabled( hknpShapeKey key, bool isEnabled ) HK_OVERRIDE;
			virtual bool commitChanges() HK_OVERRIDE;
#endif
			virtual bool isShapeKeyEnabled( hknpShapeKey key ) const HK_OVERRIDE;
			virtual int calcSize() const HK_OVERRIDE;

			hknpMaskedCompositeShape* m_maskedShape;
		};

	public:	

		HK_ON_SPU( mutable ) hkRefPtr<const hknpCompositeShape> m_shape;
		hknpShapeKeyMask*	m_mask;
		MaskWrapper			m_maskWrapper;
		MutationSignals		m_mutationSignals;

		/// The size of the masked composite shape. This value is only valid and used on PlayStation(R)3.
		/// On SPU this value will be reset to 0 once the child shape has been uploaded to SPU.
		mutable int m_childShapeSize;

		/// The size of the mask itself. This value is only valid and used on PlayStation(R)3.
		mutable int m_maskSize;
};

#include <Physics/Physics/Collide/Shape/Composite/Masked/hknpMaskedCompositeShape.inl>

#endif // HKNP_MASKED_COMPOSITE_SHAPE_H

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

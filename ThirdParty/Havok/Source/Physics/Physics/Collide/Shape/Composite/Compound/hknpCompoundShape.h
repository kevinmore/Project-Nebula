/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COMPOUND_SHAPE_H
#define HKNP_COMPOUND_SHAPE_H

#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpShapeInstance.h>

#include <Common/Base/Container/FreeListArray/hkFreeListArray.h>
#include <Common/Base/Types/hkHandle.h>

extern const hkClass hknpCompoundShapeClass;


/// Base class for compound shapes which contain a list of child shape instances.
/// The child shapes can be of any type (even hknpCompoundShape).
/// Note: If you use a hknpHeightFieldShape as a child, the height field shape will collide as composite (not SDF).
class hknpCompoundShape : public hknpCompositeShape
{
	//+version(1)

	public:

		typedef hkFreeListArray<hknpShapeInstance, hknpShapeInstanceId, 8, hknpShapeInstance> InstanceFreeListArray;

	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		HK_DECLARE_REFLECTION();

		/// Get the maximum number of shape instances allowed.
		HK_FORCE_INLINE int getCapacity() const;

		/// Get all allocated instance IDs.
		HK_FORCE_INLINE void getAllInstanceIds( hkArray<hknpShapeInstanceId>& instancesIdsOut ) const;

		/// Get an iterator over all allocated shape instances.
		HK_FORCE_INLINE InstanceFreeListArray::Iterator getShapeInstanceIterator() const;

		/// Get an instance ID from a shape key.
		HK_FORCE_INLINE hknpShapeInstanceId getInstanceIdFromShapeKey( hknpShapeKey key ) const;

		/// Get read-only access to an instance.
		HK_FORCE_INLINE const hknpShapeInstance& getInstance( hknpShapeInstanceId instance ) const;

		/// Enables or disables an instance.
		
		HK_FORCE_INLINE void setInstanceEnabled( hknpShapeInstanceId instanceId, bool isEnabled );

		/// Set an instance's shape tag.
		HK_FORCE_INLINE void setInstanceShapeTag( hknpShapeInstanceId instanceId, hknpShapeTag tag );

		/// Update the local space AABB, which should enclose all enabled instances.
		/// This is called automatically whenever instances are changed.
		/// Returns true if the AABB changed.
		virtual bool updateAabb() = 0;

		/// Fills the collector with the immediate child shape as identified by the shape \a key and returns the
		/// shortened shape key (to be used for further descending down the shape hierarchy.)
		/// Returns HKNP_INVALID_SHAPE_KEY if the shape referenced by \a key was a leaf shape.
		virtual hknpShapeKey getChildShape( hknpShapeKey key, hknpShapeCollector* collector ) const = 0;

		//
		// hknpShape implementation
		//

#if defined( HK_PLATFORM_SPU )

		virtual hknpShapeKeyIterator* createShapeKeyIterator(
			hkUint8* buffer, int bufferSize, const hknpShapeKeyMask* mask = HK_NULL ) const HK_OVERRIDE;
#else

		virtual MutationSignals* getMutationSignals() HK_OVERRIDE;

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

#if !defined(HK_PLATFORM_SPU)

		virtual hkRefNew<hknpShapeKeyIterator> createShapeKeyIterator(
			const hknpShapeKeyMask* mask = HK_NULL ) const HK_OVERRIDE;

		virtual void buildMassProperties(
			const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const HK_OVERRIDE;

		virtual hkResult buildSurfaceGeometry(
			const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const HK_OVERRIDE;

		virtual void checkConsistency() const HK_OVERRIDE;

		/// hkReferencedObject implementation
		virtual const hkClass* getClassType() const HK_OVERRIDE;

#if defined( HK_PLATFORM_HAS_SPU )

		/// This helper method traverses the shape tree and propagates all SPU flags upwards to the root.
		int propagateSpuFlags(bool recomputeChildFlags = false) const;

		/// Automatically set the SPU flags on this compound shape.
		/// This method is rather slow (as it calls propagateSpuFlags()) and should only be used after a compound
		/// has been fully created or mutated.
		virtual void computeSpuFlags() HK_OVERRIDE;

#endif

#endif // !HK_PLATFORM_SPU

	protected:

#if !defined( HK_PLATFORM_SPU )

		/// Constructor.
		HK_FORCE_INLINE hknpCompoundShape( bool isMutable );

		/// Serialization constructor.
		HK_FORCE_INLINE hknpCompoundShape( hkFinishLoadedObjectFlag flag );

		/// Calculate the number of shape key bits required for a maximum number of instances.
		HK_FORCE_INLINE static hkUint8 calcNumShapeKeyBits( int maxNumInstances );

#endif // !HK_PLATFORM_SPU

		HKNP_DECLARE_SHAPE_VTABLE_UTIL_CONSTRUCTOR( hknpCompoundShape, hknpCompositeShape );

		/// Destructor.
		virtual ~hknpCompoundShape();

	protected:

		/// Free list array of child shape instances. Do not use this directly.
		HK_ALIGN16( InstanceFreeListArray m_instances );

		/// Whether this compound can be changed at runtime (i.e. whether the engine should listen to m_mutationSignals).
		hkBool m_isMutable;

		/// A signal which is fired whenever any of the child shapes are mutated.
		MutationSignals m_mutationSignals;	//+nosave +overridetype(void*)

		/// A cached AABB of all enabled child shape instances.
		hkAabb m_aabb;	//+nosave
};

typedef hknpCompoundShape::InstanceFreeListArray::Iterator hknpShapeInstanceIterator;

#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpCompoundShape.inl>

#endif // HKNP_COMPOUND_SHAPE_H

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

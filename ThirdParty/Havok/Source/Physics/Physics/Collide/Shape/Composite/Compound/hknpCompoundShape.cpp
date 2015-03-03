/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpCompoundShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Scaled/hknpScaledConvexShape.h>

#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>

namespace
{
	/// An iterator to enumerate all enabled keys in a compound shape.
	class hknpCompoundShapeKeyIterator : public hknpShapeKeyIterator
	{
		public:

			HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
			virtual void next() HK_OVERRIDE;

#if !defined(HK_PLATFORM_SPU)
			HK_FORCE_INLINE hknpCompoundShapeKeyIterator( const hknpShape& shape, const hknpShapeKeyMask* mask );
#else
			HK_FORCE_INLINE hknpCompoundShapeKeyIterator( hkUint8* subIteratorBuffer, int subIteratorBufferSize,
				const hknpShape& shape, const hknpShapeKeyMask* mask );
#endif
			HK_FORCE_INLINE void updateKeyFromInstanceIndex( int startingInstanceIndex );


#if !defined(HK_PLATFORM_SPU)
			hkRefPtr<hknpShapeKeyIterator> m_subIterator;
#else
			hknpShapeKeyIterator* m_subIterator;
			hkUint8* m_subIteratorBuffer;
			int m_subIteratorBufferSize;
#endif

	};

	HK_FORCE_INLINE hknpCompoundShapeKeyIterator::hknpCompoundShapeKeyIterator(
#if defined(HK_PLATFORM_SPU)
		hkUint8* subIteratorBuffer, int subIteratorBufferSize,
#endif
		const hknpShape& shape, const hknpShapeKeyMask* mask )
	:	hknpShapeKeyIterator( shape, mask )
#if defined(HK_PLATFORM_SPU)
	,	m_subIteratorBuffer(subIteratorBuffer)
	,	m_subIteratorBufferSize(subIteratorBufferSize)
#endif
	{
		HK_ASSERT( 0x5463643b,
			m_shape->getType() == hknpShapeType::STATIC_COMPOUND ||
			m_shape->getType() == hknpShapeType::DYNAMIC_COMPOUND );
		updateKeyFromInstanceIndex(0);
	}

	void hknpCompoundShapeKeyIterator::next()
	{
		if( isValid() )
		{
			const hknpCompoundShape* cs = (const hknpCompoundShape*)m_shape;

			// Get the index of the instance to which the keyPath parameter points to. Also get the associated child shape.
			const int instanceIndex = int( m_keyPath.getKey() >> HKNP_NUM_UNUSED_SHAPE_KEY_BITS(cs->getNumShapeKeyBits()) );

			// Get the next key in the current instance.
			m_subIterator->next();

			if( m_subIterator->isValid() )
			{
				// Left-align local key and append valid subtree key to it.
				// Note that subTreeKeyPath already has all unused bits set to 1.
				hknpShapeKey key = hknpShapeKey( (instanceIndex << HKNP_NUM_UNUSED_SHAPE_KEY_BITS(cs->getNumShapeKeyBits())) | (m_subIterator->getKey() >> cs->getNumShapeKeyBits()) );
				m_keyPath.setFromKey( key, cs->getNumShapeKeyBits() + m_subIterator->getKeyPath().getKeySize() );
			}
			else
			{
				// So we hit the last key in the current instance.
				// Search for the first valid key in the remaining instances, if available.
				updateKeyFromInstanceIndex( instanceIndex + 1 );
			}
		}
	}

	HK_FORCE_INLINE void hknpCompoundShapeKeyIterator::updateKeyFromInstanceIndex( int startingInstanceIndex )
	{
		const hknpCompoundShape* cs = (const hknpCompoundShape*)m_shape;

		for( hknpShapeInstanceIterator it = cs->getShapeInstanceIterator(); it.isValid(); it.next() )
		{
			if( it.getIndex().value() < startingInstanceIndex )
			{
				continue;
			}

			// On SPU the array of instances is not downloaded, so we must call getInstance and DMA them from main memory one by one
			const hknpShapeInstanceId instanceId = it.getIndex();
			const hknpShapeInstance& instance = cs->getInstance(instanceId);
			if( instance.getFlags() & hknpShapeInstance::IS_ENABLED )
			{
				const int instanceIndex = instanceId.value();

				// Left-align local instance key. +1 before shift, -1 afterwards forces all unused bits to 1.
				hknpShapeKey instanceKey = hknpShapeKey( ((instanceIndex+1) << HKNP_NUM_UNUSED_SHAPE_KEY_BITS(cs->getNumShapeKeyBits())) - 1);

				if( !m_mask || m_mask->isShapeKeyEnabled(instanceKey) )
				{
#if !defined(HK_PLATFORM_SPU)
					m_subIterator = instance.getShape()->createShapeKeyIterator( HK_NULL );
#else
					const hknpShape* childShapePpu = instance.getShape();
					hknpShape* childShapeSpu = reinterpret_cast<hknpShape*>(m_subIteratorBuffer);

					// Reserve space for the shape on Spu
					const int childShapeSize = hkMath::min2(instance.getShapeMemorySize(), HKNP_MAX_SIZEOF_SHAPE);
					const int bufferSizeLeft = m_subIteratorBufferSize - childShapeSize;
					HK_ASSERT(0x6dc8cece, bufferSizeLeft >= 0);
					hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(childShapeSpu, childShapePpu, childShapeSize, hkSpuDmaManager::READ_COPY);
					HK_SPU_DMA_PERFORM_FINAL_CHECKS( childShapePpu, childShapeSpu, childShapeSize );
					hknpShapeVirtualTableUtil::patchVirtualTable(childShapeSpu);

					// Create the sub-iterator
					m_subIterator = childShapeSpu->createShapeKeyIterator( &m_subIteratorBuffer[childShapeSize], bufferSizeLeft, HK_NULL );
#endif

					// Left-align local instance key and append valid subtree key to it.
					// Note that subTreeKey already has all unused bits set to 1.
					hknpShapeKey key = hknpShapeKey((instanceIndex << HKNP_NUM_UNUSED_SHAPE_KEY_BITS(cs->getNumShapeKeyBits())) | (m_subIterator->getKey() >> cs->getNumShapeKeyBits()));
					m_keyPath.setFromKey( key, cs->getNumShapeKeyBits() + m_subIterator->getKeyPath().getKeySize() );
					return;
				}
			}
		}

		m_subIterator = HK_NULL;
		m_keyPath.reset();
	}

}	// anonymous namespace

#if !defined(HK_PLATFORM_SPU)

hkRefNew<hknpShapeKeyIterator> hknpCompoundShape::createShapeKeyIterator( const hknpShapeKeyMask* mask ) const
{
	return hkRefNew<hknpShapeKeyIterator>( new hknpCompoundShapeKeyIterator( *this, mask ) );
}

#else

hknpShapeKeyIterator* hknpCompoundShape::createShapeKeyIterator( hkUint8* buffer, int bufferSize, const hknpShapeKeyMask* mask ) const
{
	int iteratorSize = HK_NEXT_MULTIPLE_OF(16, sizeof(hknpCompoundShapeKeyIterator));

	HK_ASSERT( 0xaf1fe143, iteratorSize < bufferSize );

	hkUint8* subBuffer = hkAddByteOffset( buffer, iteratorSize );
	int subBufferSize = bufferSize - iteratorSize;

	hknpCompoundShapeKeyIterator* it = new (buffer) hknpCompoundShapeKeyIterator( subBuffer, subBufferSize, *this, mask );
	return it;
}

#endif

#if !defined(HK_PLATFORM_SPU)

hknpCompoundShape::~hknpCompoundShape()
{
	m_mutationSignals.m_shapeDestroyed.fire();
}

hknpShape::MutationSignals* hknpCompoundShape::getMutationSignals()
{
	if( m_isMutable )
	{
		return &m_mutationSignals;
	}
	return HK_NULL;
}

#if defined( HK_PLATFORM_HAS_SPU )

int hknpCompoundShape::propagateSpuFlags(bool recomputeChildFlags) const
{
	int spuFlags = 0;

	for (int i = 0, n = m_instances.getCapacity(); i < n; i++)
	{
		const hknpShapeInstanceId instanceId(i);
		if (m_instances.isAllocated(instanceId))
		{
			const hknpShapeInstance& instance = getInstance(instanceId);
			if (instance.getFlags() & hknpShapeInstance::IS_ENABLED)
			{
				const hknpShape* shape = instance.getShape();

				// Make sure the child shape has computed its flags
				if ( recomputeChildFlags )
				{
					const_cast<hknpShape*>(shape)->computeSpuFlags();
				}

				if ( shape->getType() == hknpShapeType::STATIC_COMPOUND || shape->getType() == hknpShapeType::DYNAMIC_COMPOUND )
				{
					const hknpCompoundShape* compound = (const hknpCompoundShape*)shape;
					spuFlags |= compound->propagateSpuFlags();
				}
				else
				{
					spuFlags |= ( shape->getFlags().get( hknpShape::NO_GET_ALL_SHAPE_KEYS_ON_SPU ) | shape->getFlags().get( hknpShape::SHAPE_NOT_SUPPORTED_ON_SPU ) );
				}
			}
		}
	}

	return spuFlags;
}

void hknpCompoundShape::computeSpuFlags()
{
	m_flags.orWith( hkUint16(propagateSpuFlags(true)) );
}

#endif

#endif

#if !defined(HK_PLATFORM_SPU)
void hknpCompoundShape::getAllShapeKeys(
	const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,
	hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const
#else
void hknpCompoundShape::getAllShapeKeys(
	const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask,
	hkUint8* shapeBuffer, int shapeBufferSize,
	hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const
#endif
{
	for (int i = 0, n = m_instances.getCapacity(); i < n; i++)
	{
		const hknpShapeInstanceId instanceId(i);
		if (m_instances.isAllocated(instanceId))
		{
			const hknpShapeInstance& instance = getInstance(instanceId);
			if (instance.getFlags() & hknpShapeInstance::IS_ENABLED)
			{
				// Left-align local instance key. +1 before shift, -1 afterwards forces all unused bits to 1.
				hknpShapeKey instanceKey = hknpShapeKey(((i+1) << HKNP_NUM_UNUSED_SHAPE_KEY_BITS(m_numShapeKeyBits)) - 1);

				if (!mask || mask->isShapeKeyEnabled(instanceKey))
				{
					hknpShapeKeyPath childPath = shapeKeyPath;
					childPath.appendSubKey(i, m_numShapeKeyBits);
#if !defined(HK_PLATFORM_SPU)
					instance.getShape()->getAllShapeKeys(childPath, HK_NULL, keyPathsOut);
#else
					int instanceShapeSize = instance.getShapeMemorySize();
					HK_ASSERT( 0xaf1e3241, instanceShapeSize <= shapeBufferSize );
					hknpShape* shapeOnSpu = reinterpret_cast<hknpShape*>(shapeBuffer);
					hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion( shapeOnSpu, instance.getShape(), instanceShapeSize, hkSpuDmaManager::READ_COPY );
					hkSpuDmaManager::performFinalChecks( instance.getShape(), shapeOnSpu, instance.getShapeMemorySize() );
					hknpShapeVirtualTableUtil::patchVirtualTable( shapeOnSpu );
					hkUint8* newShapeBuffer = hkAddByteOffset( shapeBuffer, instanceShapeSize );
					int newShapeBufferSize = shapeBufferSize - instanceShapeSize;
					shapeOnSpu->getAllShapeKeys( childPath, HK_NULL, newShapeBuffer, newShapeBufferSize, keyPathsOut );
#endif
				}
			}
		}
	}
}


#if !defined(HK_PLATFORM_SPU)

void hknpCompoundShape::buildMassProperties(
	const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const
{
	switch( massConfig.m_quality )
	{
	case MassConfig::QUALITY_LOW:
		{
			// Calculate using root shape AABB
			hknpShape::buildMassProperties( massConfig, massPropertiesOut );
		}
		break;

	case MassConfig::QUALITY_MEDIUM:
	case MassConfig::QUALITY_HIGH:
		{
			// Gather mass elements for each enabled instance.
			// If the instance shapes have mass properties use them, otherwise build them using massConfig.
			const MassConfig& childConfig = massConfig;
			hkLocalArray<hkMassElement> massElements( m_instances.getCapacity() );
			for( hknpShapeInstanceIterator it = getShapeInstanceIterator(); it.isValid(); it.next() )
			{
				const hknpShapeInstance& instance = it.getValue();
				if( instance.isEnabled() )
				{
					hkDiagonalizedMassProperties dmp;
					const hknpShapeMassProperties* props = (const hknpShapeMassProperties*)(
						instance.getShape()->getProperty( hknpShapePropertyKeys::MASS_PROPERTIES ) );
					if( props )
					{
						props->m_compressedMassProperties.unpack( &dmp );
					}
					else
					{
						instance.getShape()->buildMassProperties( childConfig, dmp );
					}

					hkMassElement& me = massElements.expandOne();
					dmp.unpack( &me.m_properties );
					instance.getFullTransform( me.m_transform );
				}
			}

			// Combine them
			hkMassProperties combinedMassProperties;
			hkInertiaTensorComputer::combineMassProperties( massElements, combinedMassProperties );
			massPropertiesOut.pack( combinedMassProperties );

			/*
			Currently we don't scale to the requested mass/density. We assume that the argument is a density that applies to the children
			that don't have mass properties. _IF_ we were to rescale it, we should do this:
			hkReal massMultiplier = massConfig.calcMassFromVolume( massPropertiesOut.m_volume )/massPropertiesOut.m_mass;
			massPropertiesOut.m_mass *= massMultiplier;
			massPropertiesOut.m_inertiaTensor.mul( hkSimdReal::fromFloat(massMultiplier) );
			*/
		}
		break;

	default:
		HK_ASSERT( 0x6b809a4c, !"Should not get here" );
		break;
	}
}

hkResult hknpCompoundShape::buildSurfaceGeometry(
	const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const
{
	geometryOut->clear();

	// Append geometry for each enabled instance
	for( int i=0, n=m_instances.getCapacity(); i<n; ++i )
	{
		const hknpShapeInstanceId instanceId(i);
		if( m_instances.isAllocated( instanceId ) )
		{
			const hknpShapeInstance& instance = getInstance( instanceId );
			if( instance.getFlags() & hknpShapeInstance::IS_ENABLED )
			{
				// If the instance shape is convex and we have scale we need to wrap it in a scaled convex shape
				const hknpShape* instanceShape = instance.getShape();
				const hknpShapeType::Enum instanceType = instanceShape->getType();
				HK_ALIGN16(hkUint8 scaledShapeBuffer[sizeof(hknpScaledConvexShapeBase)]);
				hkTransform transform;
				if (!(instance.getFlags() & hknpShapeInstance::HAS_SCALE) ||
					(instanceType != hknpShapeType::CONVEX && instanceType != hknpShapeType::CONVEX_POLYTOPE))
				{
					instance.getFullTransform(transform);
				}
				else
				{
					instanceShape = hknpScaledConvexShapeBase::createInPlace(
						instanceShape->asConvexShape(), instance.getScale(), instance.getScaleMode(),
						scaledShapeBuffer, sizeof(hknpScaledConvexShapeBase) );
					transform = instance.getTransform();
				}

				// Build surface geometry of the instance shape without transform
				hkGeometry geometry;
				instanceShape->buildSurfaceGeometry( config.m_radiusMode, &geometry );

				// Append transformed geometry
				geometryOut->appendGeometry( geometry, transform );
			}
		}
	}

	return geometryOut->m_triangles.getSize() ? HK_SUCCESS : HK_FAILURE;
}

const hkClass* hknpCompoundShape::getClassType() const
{
	return &hknpCompoundShapeClass;
}

void hknpCompoundShape::checkConsistency() const
{
#ifdef HK_DEBUG
	// Check for any NULL child shapes
	for( hknpShapeInstanceIterator it = getShapeInstanceIterator(); it.isValid(); it.next() )
	{
		if( it.getValue().getShape() == HK_NULL )
		{
			hkStringBuf str;
			str.printf( "The compound shape at 0x%p has a NULL (unserialized?) shape in instance %i.",
				this, it.getIndex().value() );
			HK_WARN( 0xabbab1e4, str.cString() );
		}
	}
#endif
}

#endif // !defined(HK_PLATFORM_SPU)

/*
 * Havok SDK - Product file, BUILD(#20130912)
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

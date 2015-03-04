/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/hknpShapeUtil.h>

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Capsule/hknpCapsuleShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Scaled/hknpScaledConvexShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpExternMeshShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/Static/hknpStaticCompoundShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/Dynamic/hknpDynamicCompoundShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Masked/hknpMaskedCompositeShape.h>
#include <Physics/Physics/Collide/Shape/Composite/HeightField/Compressed/hknpCompressedHeightFieldShape.h>
#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>

#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>
#include <Common/Visualize/Shape/hkDisplayGeometry.h>
#include <Common/Visualize/Shape/hkDisplaySphere.h>
#include <Common/Visualize/Shape/hkDisplayCapsule.h>
#include <Common/Visualize/Shape/hkDisplayConvex.h>

#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#include <Common/Internal/GeometryProcessing/hkGeometryProcessing.h>


#if !defined(HK_PLATFORM_SPU)

namespace
{

	static void _buildShapeDisplayGeometriesRecursive(
		const hknpShape* shape, const hkTransform& transform, hkVector4Parameter scale,
		hknpShape::ConvexRadiusDisplayMode radiusMode, bool flipOrientation,
		hkArray<hkDisplayGeometry*>& displayGeometriesOut )
	{
		const hknpShapeType::Enum shapeType = shape->getType();
		switch( shapeType )
		{
		case hknpShapeType::SPHERE:
			{
				const hknpSphereShape* sphereShape = static_cast<const hknpSphereShape*>( shape);
				hkSphere sphere( sphereShape->getVertex(0), sphereShape->m_convexRadius );

				// scale assuming uniform scale
				sphere.setRadius(scale(0) * sphere.getRadius());

				hkDisplaySphere* displaySphere = new hkDisplaySphere( sphere, 10, 10 );

				displaySphere->setTransform( transform );
				displayGeometriesOut.pushBack( displaySphere );
			}
			break;

		case hknpShapeType::CAPSULE:
			{
				const hknpCapsuleShape* capsuleShape = static_cast<const hknpCapsuleShape*>( shape);
				hkVector4 a = capsuleShape->m_a;
				hkVector4 b = capsuleShape->m_b;
				hkReal radius = capsuleShape->m_convexRadius;

				// scale assuming uniform scale
				hkVector4 uniformScale; uniformScale.setAll(scale.getComponent<0>());
				a.mul(uniformScale);
				b.mul(uniformScale);
				radius *= scale(0);

				hkDisplayCapsule* displayCapsule = new hkDisplayCapsule( a, b, radius, 8 );

				displayCapsule->setTransform( transform );
				displayGeometriesOut.pushBack( displayCapsule );
			}
			break;

		case hknpShapeType::STATIC_COMPOUND:
		case hknpShapeType::DYNAMIC_COMPOUND:
			{
				const hknpCompoundShape* compound = static_cast<const hknpCompoundShape*>( shape );
				hkArray<hknpShapeInstanceId> instances; compound->getAllInstanceIds(instances);
				for( int i=0; i<instances.getSize(); ++i )
				{
					const hknpShapeInstance& instance = compound->getInstance( instances[i] );

					// Apply input scale
					hkVector4 newScale; newScale.setMul(scale, instance.getScale());

					// Scale translation by parent scale before rigid transform
					hkTransform instanceTransform = instance.getTransform();
					instanceTransform.getTranslation().mul(scale);
					hkTransform newTransform; newTransform.setMul(transform, instanceTransform);

					const bool flip = ( instance.getFlags() & hknpShapeInstance::FLIP_ORIENTATION ) ? true : false;

					// If the instance shape is convex and we have scale, we need to wrap it in a scaled convex shape
					// so that it is representative of the shape that will be created on-the-fly
					const hknpShape* instanceShape = instance.getShape();
					HK_ALIGN16( hkUint8 scaledShapeBuffer[sizeof(hknpScaledConvexShapeBase)] );
					if( instanceShape->m_dispatchType == hknpCollisionDispatchType::CONVEX &&
						!newScale.allExactlyEqual<3>( hkVector4::getConstant<HK_QUADREAL_1>() ) )
					{
						instanceShape = hknpScaledConvexShapeBase::createInPlace(
							instanceShape->asConvexShape(), newScale, instance.getScaleMode(),
							scaledShapeBuffer, sizeof(hknpScaledConvexShapeBase) );
						newScale = hkVector4::getConstant<HK_QUADREAL_1>();
					}

					// Recurse
					_buildShapeDisplayGeometriesRecursive(
						instanceShape, newTransform, newScale, radiusMode, flip, displayGeometriesOut );
				}
			}
			break;

		default:
			{
				hkGeometry*	geometry = new hkGeometry();
				if( shape->buildSurfaceGeometry(radiusMode, geometry) == HK_SUCCESS )
				{
					if( flipOrientation )
					{
						for( int i=0, num = geometry->m_triangles.getSize(); i < num; ++i )
						{
							const int temp = geometry->m_triangles[i].m_a;
							geometry->m_triangles[i].m_a = geometry->m_triangles[i].m_b;
							geometry->m_triangles[i].m_b = temp;
						}
					}

					// Apply scale
					for( int i = 0, num = geometry->m_vertices.getSize(); i < num; ++i )
					{
						geometry->m_vertices[i].mul(scale);
					}

					hkDisplayGeometry* displayGeometry = new hkDisplayConvex(geometry);
					displayGeometry->setTransform(transform);
					displayGeometriesOut.pushBack(displayGeometry);
				}
				else
				{
					delete geometry;
				}
			}
			break;
		}
	}

}	// anonymous namespace


hkResult hknpShapeUtil::buildAabbMassProperties( const hknpShape::MassConfig& massConfig,
	const hkAabb& aabb, hkDiagonalizedMassProperties& massPropertiesOut )
{
	HK_ASSERT( 0x2352ffba, aabb.isValid() );

	// Get half extents
	hkVector4 halfExtents;
	{
		halfExtents.setSub( aabb.m_max, aabb.m_min );
		halfExtents.mul( hkVector4::getConstant<HK_QUADREAL_INV_2>() );

		// Ensure that no component is zero, so that we get non-zero volume
		hkVector4 epsilon = hkVector4::getConstant( HK_QUADREAL_EPS );
		epsilon.zeroIfFalse( halfExtents.equalZero() );
		halfExtents.add( epsilon );
	}

	// Compute box volume mass properties for unit mass
	HK_ON_DEBUG( hkResult res = ) hkInertiaTensorComputer::computeBoxVolumeMassPropertiesDiagonalized(
		halfExtents, 1.0f, massPropertiesOut.m_inertiaTensor, massPropertiesOut.m_volume );
	HK_ASSERT( 0x2352ffbb, res == HK_SUCCESS );
	massPropertiesOut.m_centerOfMass.setInterpolate( aabb.m_max, aabb.m_min, hkSimdReal::getConstant<HK_QUADREAL_INV_2>() );
	massPropertiesOut.m_majorAxisSpace.setIdentity();

	// Factor in the supplied mass/density and inertia factor
	hkReal mass = massConfig.calcMassFromVolume( massPropertiesOut.m_volume );
	massPropertiesOut.m_inertiaTensor.mul( hkSimdReal::fromFloat( mass*massConfig.m_inertiaFactor ) );
	massPropertiesOut.m_mass = mass;

	return HK_SUCCESS;
}

void HK_CALL hknpShapeUtil::buildShapeDisplayGeometries(
	const hknpShape* shape, const hkTransform& transform, hkVector4Parameter scale,
	hknpShape::ConvexRadiusDisplayMode radiusMode, hkArray<hkDisplayGeometry*>& displayGeometriesOut )
{
	_buildShapeDisplayGeometriesRecursive( shape, transform, scale, radiusMode, false, displayGeometriesOut );
}

hkResult HK_CALL hknpShapeUtil::createConvexHullGeometry(
	const hknpShape& shape, hknpShape::ConvexRadiusDisplayMode radiusMode,
	hkGeometry* geometryInOut, int material )
{
	// Use shape faces if present
	const int numPlanes = shape.getNumberOfFaces();
	if( (numPlanes > 0) && ( (radiusMode != hknpShape::CONVEX_RADIUS_DISPLAY_ROUNDED) || (shape.m_convexRadius == 0.0f) ) )
	{
		hkArray<hkVector4>::Temp expandedPlanes( numPlanes );
		for( int i=0; i<numPlanes; ++i )
		{
			hkVector4& plane = expandedPlanes[i];
			int minAngle; shape.getFaceInfo( i, plane, minAngle );
			if( radiusMode == hknpShape::CONVEX_RADIUS_DISPLAY_PLANAR )
			{
				plane(3) -= shape.m_convexRadius;
			}
		}

		hkgpConvexHull hull;
		hkgpConvexHull::BuildConfig config;
		config.m_allowLowerDimensions = true;
		if( hull.buildFromPlanes( expandedPlanes.begin(), expandedPlanes.getSize() ) != -1 )
		{
			hull.generateGeometry( hkgpConvexHull::SOURCE_VERTICES, *geometryInOut, material );
			return HK_SUCCESS;
		}
	}

	// Use shape vertices otherwise
	const int numVertices = shape.getNumberOfSupportVertices();
	if( numVertices > 0 )
	{
		hkArray<hkcdVertex>::Temp verticesBuffer;
		verticesBuffer.setSize( numVertices );
		const hkcdVertex* vertices = shape.getSupportVertices( verticesBuffer.begin(), numVertices );

		return createConvexHullGeometry( vertices, numVertices, shape.m_convexRadius, radiusMode, geometryInOut, material );
	}

	return HK_FAILURE;
}

namespace
{
	// A set of normals used to expand convex hull vertices
	template<int NUM_SAMPLES>
	class ExpansionNormals : public hkInplaceArray< hkVector4, NUM_SAMPLES*NUM_SAMPLES >
	{
		public:

			ExpansionNormals()
			{
				this->reserve( NUM_SAMPLES * NUM_SAMPLES );
				const hkReal invNs = 1.0f / (NUM_SAMPLES - 1);
				for( int u=0; u<NUM_SAMPLES; ++u )
				{
					hkVector4 uv;
					uv(0) = u * invNs;
					for( int v=0; v<NUM_SAMPLES; ++v )
					{
						uv(1) = v * invNs;
						hkGeometryProcessing::octahedronToNormal( uv, this->expandOne() );
					}
				}
			}
	};
}

hkResult HK_CALL hknpShapeUtil::createConvexHullGeometry(
	const hkVector4* vertices, int numVertices, hkReal convexRadius, hknpShape::ConvexRadiusDisplayMode radiusMode,
	hkGeometry* geometryInOut, int material )
{
	// Prepare vertices
	hkArray<hkVector4>::Temp expandedVertices;
	{
		if( radiusMode == hknpShape::CONVEX_RADIUS_DISPLAY_ROUNDED && convexRadius > 0.0f )
		{
			// Turn each vertex into a set of expanded vertices
			static ExpansionNormals<7> normals;
			const int numNormals = normals.getSize();
			expandedVertices.reserve( numVertices * numNormals );
			hkSimdReal radius; radius.setFromFloat( convexRadius );
			for( int i=0; i<numVertices; ++i )
			{
				hkVector4* HK_RESTRICT v = expandedVertices.expandBy( numNormals );
				for( int j=0; j<numNormals; ++j )
				{
					v[j].setAddMul( vertices[i], normals[j], radius );
				}
			}
		}
		else
		{
			expandedVertices.setDataUserFree( const_cast<hkVector4*>(vertices), numVertices, numVertices );
		}
	}

	// Build a hull around all the vertices
	hkgpConvexHull hull;
	hkgpConvexHull::BuildConfig config;
	config.m_allowLowerDimensions = true;
	if( hull.build( expandedVertices, config ) == -1 )
	{
		return HK_FAILURE;
	}

	if( radiusMode == hknpShape::CONVEX_RADIUS_DISPLAY_PLANAR && convexRadius > 0.0f )
	{
		hull.expandByPlanarMargin( convexRadius );
	}

	hull.generateGeometry( hkgpConvexHull::SOURCE_VERTICES, *geometryInOut, material );
	return HK_SUCCESS;
}

#endif // !defined(HK_PLATFORM_SPU)

void HK_CALL hknpShapeUtil::calcScalingParameters(
	const hknpConvexShape& shape, hknpShape::ScaleMode mode,
	hkVector4* HK_RESTRICT scaleInOut, hkReal* HK_RESTRICT radiusInOut, hkVector4* HK_RESTRICT offsetOut )
{
	// Clamp the scale to safe limits
	hkVector4 minScale; minScale.setAll( HKNP_SHAPE_MIN_SCALE );
	hkVector4 scale; scale.setMax(*scaleInOut, minScale);

	// If the shape is a sphere or a capsule scale the radius directly regardless of the scale mode
	const hknpShapeType::Enum shapeType = shape.getType();
	if (shapeType == hknpShapeType::SPHERE || shapeType == hknpShapeType::CAPSULE)
	{
		HK_ON_DEBUG(hkVector4 scaleX; scaleX.setAll(scale.getComponent<0>()));
		HK_WARN_ON_DEBUG_IF(!scale.allExactlyEqual<3>(scaleX), 0x22c65a7b, "Non-uniform scale cannot be applied to \
			spheres or capsules. Uniform scaling will be applied using X component");
		*radiusInOut *= scale(0);
		offsetOut->setZero();
	}

	// Apply surface scaling when required computing a scale and a translation such that the child shape's AABB ends
	// up being scaled by the user provided scale, taking the convex radius into account.
	else if ((mode == hknpShape::SCALE_SURFACE) && !hkMath::equal(*radiusInOut, 0.0f) &&
			 !scale.allEqual<3>(hkVector4::getConstant<HK_QUADREAL_1>(), hkSimdReal_Eps))
	{
		// Calculate the AABB of the child shape vertices (excluding the convex radius)
		hkVector4 halfExtents;
		hkVector4 center;
		{
			const hkcdVertex* vertices = shape.getVertices();
			const int numVertices = shape.getNumberOfVertices();
			hkAabb aabb; aabb.setEmpty();
			for (int i = 0; i < numVertices; i += 4, vertices += 4)
			{
				hkMxVector<4> vertsMx; vertsMx.moveLoad(vertices);
				hkMxUNROLL_4(aabb.includePoint(vertsMx.getVector<hkMxI>()));
			}
			aabb.getHalfExtents(halfExtents);
			aabb.getCenter(center);
		}

		// Clamp the radius such that it fits inside the scaled shape
		hkVector4 childRadiusVector; childRadiusVector.setAll(*radiusInOut);
		hkVector4 scaleAbs; scaleAbs.setAbs(scale);
		hkVector4 clampedRadius;
		{
			// Calculate the maximum radius the scale allows for each component
			// scaleAbs * (halfExtent + childRadius)
			clampedRadius.setAdd(halfExtents, childRadiusVector);
			clampedRadius.mul(scaleAbs);

			// If the child radius is greater than the maximum allowed for any component,
			// use the maximum allowed for all components.
			if (!childRadiusVector.allLess<3>(clampedRadius))
			{
				clampedRadius.setHorizontalMin<3>(clampedRadius);
				HK_WARN_ONCE(0x1592e03a, "The convex radius has been reduced to fit the scaled shape");
			}
			else
			{
				clampedRadius = childRadiusVector;
			}
		}

		// Compute the extra scale
		hkVector4 additionalScale;
		scaleAbs.mul(childRadiusVector);
		scaleAbs.sub(clampedRadius);
		additionalScale.setDiv<HK_ACC_23_BIT, HK_DIV_SET_ZERO>(scaleAbs, halfExtents);
		scale.add(additionalScale);

		// Clamp scale to safe limits
		scale.setMax(scale, minScale);

		*radiusInOut = clampedRadius(0);
		offsetOut->setMul(additionalScale, center);
	}

	// Just scale vertices
	else
	{
		offsetOut->setZero();
	}

	scale.setW(*scaleInOut);
	*scaleInOut = scale;
	HK_ASSERT2(0xf0235465, scaleInOut->greaterZero().allAreSet(hkVector4ComparisonMask::MASK_XYZ),
		"Scale must be greater than zero in all dimensions" );
}

#if !defined(HK_PLATFORM_SPU)

void HK_CALL hknpShapeUtil::flattenIntoConvexShapes(
	const hknpShape* shape, const hkTransform& worldFromParent,
	hkArray<const hknpConvexShape*>& shapesOut, hkArray<hkTransform>& worldFromShapesOut )
{
	const hknpConvexShape* convexShape = shape->asConvexShape();
	if( convexShape )
	{
		// The shape itself is a convex
		shapesOut.pushBack(convexShape);
		worldFromShapesOut.pushBack(worldFromParent);
	}
	else
	{
		// Composite shape
		hknpShapeCollectorWithInplaceTriangle childShapeCollector;
		for( hkRefPtr<hknpShapeKeyIterator> it = shape->createShapeKeyIterator(); it->isValid(); it->next() )
		{
			// Get the leaf convex shape
			childShapeCollector.reset(worldFromParent);
			shape->getLeafShape(it->getKey(), &childShapeCollector);
			convexShape = childShapeCollector.m_shapeOut->asConvexShape();
			HK_ASSERT(0x1dcd100b, convexShape);

			// Push it
			shapesOut.pushBack(convexShape);
			worldFromShapesOut.pushBack(childShapeCollector.m_transformOut);
		}
	}
}

#endif

namespace hknpShapeUtilImpl
{
#define Max2(a, b)												(((a) > (b)) ? (a) : (b))
#define Max3(a, b, c)											Max2(Max2(a, b), c)
#define Max4(a, b, c, d)										Max2(Max2(a, b), Max2(c, d))
#define Max7(a, b, c, d, e, f, g)								Max2(Max4(a, b, c, d), Max3(e, f, g))
#define Max8(a, b, c, d, e, f, g, h)							Max2(Max4(a, b, c, d), Max4(e, f, g, h))
#define Max15(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o)		Max2(Max8(a, b, c, d, e, f, g, h), Max7(i, j, k, l, m, n, o))

	/// An iterator to enumerate all enabled keys in a compound shape.
	struct MaxShapeSize
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpShapeUtilImpl::MaxShapeSize );

		enum
		{
			MAX_UNALIGNED_SHAPE_SIZE		= Max15(sizeof(hknpCapsuleShape),				sizeof(hknpConvexPolytopeShape),\
													sizeof(hknpScaledConvexShape),			sizeof(hknpSphereShape),\
													sizeof(hknpTriangleShape),				sizeof(hknpConvexShape),\
													sizeof(hknpStaticCompoundShape),		sizeof(hknpDynamicCompoundShape),\
													sizeof(hknpCompressedMeshShape),		sizeof(hknpExternMeshShape),\
													sizeof(hknpCompressedHeightFieldShape), sizeof(hknpMaskedCompositeShape),\
													sizeof(hknpCompoundShape),				sizeof(hknpCompositeShape),
													sizeof(hknpHeightFieldShape)),
			MAX_ALIGNED_SHAPE_SIZE			= HK_NEXT_MULTIPLE_OF(16, MAX_UNALIGNED_SHAPE_SIZE),
		};
	};

	HK_COMPILE_TIME_ASSERT(HKNP_MAX_SIZEOF_SHAPE >= MaxShapeSize::MAX_ALIGNED_SHAPE_SIZE);
}

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

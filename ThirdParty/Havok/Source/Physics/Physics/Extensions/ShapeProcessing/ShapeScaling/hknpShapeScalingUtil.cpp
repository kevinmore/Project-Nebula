/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/ShapeProcessing/ShapeScaling/hknpShapeScalingUtil.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/Static/hknpStaticCompoundShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/Dynamic/hknpDynamicCompoundShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Masked/hknpMaskedCompositeShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Capsule/hknpCapsuleShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>

#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Math/Matrix/hkMatrix4Util.h>

//
//	Utility functions

namespace hknpShapeScalingUtilImpl
{
	//
	//	Types

	typedef hknpShapeScalingUtil::ShapePair ShapePair;
	static hknpShape* HK_CALL scaleShape(const hkMatrix4& worldFromShape, const hknpShape* shape, hkVector4Parameter vScale, hkArray<ShapePair>& doneShapes);

	//
	//	Computes the scaling matrix

	static void HK_CALL computeScalingMatrix(const hkMatrix4& worldFromShape, hkVector4Parameter vScale, hkMatrix4& scaleMatrixOut)
	{
		hkVector4 vDiag;					vDiag.setXYZ_W(vScale, hkSimdReal_1);
		hkMatrix4 diag;						diag.setDiagonal(vDiag);
		hkMatrix4 shapeFromWorld;			hkMatrix4Util::setInverse(worldFromShape, shapeFromWorld, hkSimdReal_Eps);
		hkMatrix4 scaledWorldFromShape;		scaledWorldFromShape.setMul(diag, worldFromShape);

		scaleMatrixOut.setMul(shapeFromWorld, scaledWorldFromShape);
	}

	//
	//	Scales a static compound

	static hknpShape* HK_CALL scaleCompoundShape(const hkMatrix4& worldFromShape, const hknpCompoundShape* scs, hkVector4Parameter vScale, hkArray<ShapePair>& doneShapes, bool createDynamic)
	{
		hkMatrix4 scaleMatrix;
		computeScalingMatrix(worldFromShape, vScale, scaleMatrix);

		hkArray<hknpShapeInstance> scaledInstances;
		for (hknpShapeInstanceIterator it = scs->getShapeInstanceIterator(); it.isValid(); it.next())
		{
			// Create the scaled instance
			const hknpShapeInstance& srcInst	= it.getValue();
			hknpShapeInstance dstInst			= srcInst;

			// Get the child transform
			hkTransform tm;					dstInst.getFullTransform(tm);
			hkMatrix4 shapeFromChild;		shapeFromChild.set(tm);
			hkMatrix4 worldFromChild;		worldFromChild.setMul(worldFromShape, shapeFromChild);

			// Scale the translation
			{
				tm = dstInst.getTransform();
				scaleMatrix.transformDirection(tm.getTranslation(), tm.getTranslation());
				dstInst.setTransform(tm);
			}

			// Scale the child shape
			const hknpShape* srcChildShape	= dstInst.getShape();
			hknpShape* scaledChildShape		= scaleShape(worldFromChild, srcChildShape, vScale, doneShapes);
			if ( scaledChildShape )
			{
				dstInst.setShape(scaledChildShape);
				scaledInstances.pushBack(dstInst);
			}
		}

		// Create the shape
		hknpShape::MassConfig massConfig = hknpShape::MassConfig::fromMass(1.0f, 1.0f, hknpShape::MassConfig::QUALITY_HIGH);
		if ( createDynamic )
		{
			return new hknpDynamicCompoundShape(scaledInstances.begin(), scaledInstances.getSize(), scaledInstances.getSize(), &massConfig);
		}

		return new hknpStaticCompoundShape(scaledInstances.begin(), scaledInstances.getSize(), &massConfig, scs->isMutable());
	}

	//
	//	Scales a convex shape

	static hknpShape* HK_CALL scaleConvexShape(const hkMatrix4& worldFromShape, const hknpConvexShape* cvx, hkVector4Parameter vScale)
	{
		hkMatrix4 scaleMatrix;
		computeScalingMatrix(worldFromShape, vScale, scaleMatrix);

		// Build the surface geometry, in order to expand the vertices by the unscaled convex radius
		hknpShape::BuildSurfaceGeometryConfig geomCfg;
		geomCfg.m_radiusMode = hknpShape::CONVEX_RADIUS_DISPLAY_PLANAR;
		hkGeometry geom; cvx->buildSurfaceGeometry(geomCfg, &geom);

		// Scale vertices
		const int numVerts = geom.m_vertices.getSize();
		hkLocalBuffer<hkVector4> scaledVerts(numVerts);
		for (int i = 0; i < numVerts; i++)
		{
			scaleMatrix.transformDirection(geom.m_vertices[i], scaledVerts[i]);
		}

		// Use an average scaled convex radius
		hkSimdReal avgScale;	avgScale.setMul(hkSimdReal_Inv3, vScale.horizontalAdd<3>());
		const hkReal newRadius	= cvx->m_convexRadius * avgScale.getReal();

		// Create the shape
		hknpConvexShape::BuildConfig config;
		config.m_buildFaces				= (cvx->asConvexPolytopeShape() != HK_NULL);
		config.m_buildMassProperties	= (cvx->getProperty(hknpShapePropertyKeys::MASS_PROPERTIES) != HK_NULL);
		hkStridedVertices stridedVerts	(scaledVerts.begin(), scaledVerts.getSize());
		return hknpConvexShape::createFromVertices(stridedVerts, newRadius, config);
	}

	//
	//	Scales a sphere shape

	static hknpShape* HK_CALL scaleSphereShape(const hkMatrix4& worldFromShape, const hknpConvexShape* cvx, hkVector4Parameter vScale)
	{
		hkMatrix4 scaleMatrix;
		computeScalingMatrix(worldFromShape, vScale, scaleMatrix);

		HK_ON_DEBUG( hkVector4 scaleX; scaleX.setAll( vScale.getComponent<0>() ) );
		HK_ASSERT2( 0x17a594f6, vScale.allEqual<3>( scaleX, hkSimdReal_Eps ), "Sphere shapes do not support non-uniform scale" );

		hkVector4 vCenter;	scaleMatrix.transformDirection(cvx->getVertex(0), vCenter);
		const hkReal radius	= cvx->m_convexRadius * vScale.getComponent<0>().getReal();

		return hknpSphereShape::createSphereShape(vCenter, radius);
	}

	//
	//	Scales a capsule shape

	static hknpShape* HK_CALL scaleCapsuleShape(const hkMatrix4& worldFromShape, const hknpCapsuleShape* cvx, hkVector4Parameter vScale)
	{
		hkMatrix4 scaleMatrix;
		computeScalingMatrix(worldFromShape, vScale, scaleMatrix);

		HK_ON_DEBUG( hkVector4 scaleX; scaleX.setAll( vScale.getComponent<0>() ) );
		HK_ASSERT2( 0x17a594f6, vScale.allEqual<3>( scaleX, hkSimdReal_Eps ), "Capsule shapes do not support non-uniform scale" );

		hkVector4 vA;		scaleMatrix.transformDirection(cvx->m_a, vA);
		hkVector4 vB;		scaleMatrix.transformDirection(cvx->m_b, vB);
		const hkReal radius = cvx->m_convexRadius * vScale.getComponent<0>().getReal();

		return hknpCapsuleShape::createCapsuleShape(vA, vB, radius);
	}

	//
	//	Scales the shape

	static hknpShape* HK_CALL scaleShape(const hkMatrix4& worldFromShape, const hknpShape* shape, hkVector4Parameter vScale, hkArray<ShapePair>& doneShapes)
	{
		// To support shared shapes in a hierarchy, we check if we have done this one before.
		for (int dsi = doneShapes.getSize() - 1; dsi >= 0; dsi--)
		{
			ShapePair& p = doneShapes[dsi];

			if ( p.m_originalShape == shape )
			{
				return p.m_newShape;
			}
		}

		// Not previously scaled, do it now
		ShapePair newPair;
		newPair.m_originalShape	= shape;
		newPair.m_newShape		= HK_NULL;

		// See what to do based on the shape type
		const hknpShapeType::Enum shapeType = shape->getType();
		switch ( shapeType )
		{
		case hknpShapeType::CONVEX:
		case hknpShapeType::CONVEX_POLYTOPE:
			{
				const hknpConvexShape* oldShape	= reinterpret_cast<const hknpConvexShape*>(shape);
				hknpShape* newShape				= scaleConvexShape(worldFromShape, oldShape, vScale);
				newPair.m_newShape.setAndDontIncrementRefCount(newShape);
			}
			break;

		case hknpShapeType::SPHERE:
			{
				const hknpConvexShape* oldShape	= reinterpret_cast<const hknpConvexShape*>(shape);
				hknpShape* newShape				= scaleSphereShape(worldFromShape, oldShape, vScale);
				newPair.m_newShape.setAndDontIncrementRefCount(newShape);
			}
			break;

		case hknpShapeType::CAPSULE:
			{
				const hknpCapsuleShape* oldShape	= reinterpret_cast<const hknpCapsuleShape*>(shape);
				hknpShape* newShape				= scaleCapsuleShape(worldFromShape, oldShape, vScale);
				newPair.m_newShape.setAndDontIncrementRefCount(newShape);
			}
			break;

		case hknpShapeType::STATIC_COMPOUND:
			{
				const hknpCompoundShape* oldShape	= reinterpret_cast<const hknpCompoundShape*>(shape);
				hknpShape* newShape					= scaleCompoundShape(worldFromShape, oldShape, vScale, doneShapes, false);
				newPair.m_newShape.setAndDontIncrementRefCount(newShape);
			}
			break;

		case hknpShapeType::DYNAMIC_COMPOUND:
			{
				const hknpCompoundShape* oldShape	= reinterpret_cast<const hknpCompoundShape*>(shape);
				hknpShape* newShape					= scaleCompoundShape(worldFromShape, oldShape, vScale, doneShapes, true);
				newPair.m_newShape.setAndDontIncrementRefCount(newShape);
			}
			break;

		case hknpShapeType::MASKED_COMPOSITE:
			{
				const hknpMaskedCompositeShape* mks	= reinterpret_cast<const hknpMaskedCompositeShape*>(shape);
				hknpCompositeShape* newChildShape	= reinterpret_cast<hknpCompositeShape*>(scaleShape(worldFromShape, mks->m_shape, vScale, doneShapes));
				hknpMaskedCompositeShape* newShape	= new hknpMaskedCompositeShape(newChildShape);
				newShape->m_mask	= mks->m_mask;
				newPair.m_newShape.setAndDontIncrementRefCount(newShape);
			}
			break;

/*			TRIANGLE
			COMPRESSED_MESH
			EXTERN_MESH
			HEIGHT_FIELD			,
			COMPRESSED_HEIGHT_FIELD ,
			SCALED_CONVEX			,
			LOD						,*/
		default:
			{
				HK_ASSERT2( 0x408e84c7, false, "Shape not supported by hknpShapeScalingUtility" );
				break;
			}
		}

		// Return the scaled shape
		if ( newPair.m_newShape )
		{
			doneShapes.pushBack(newPair);
		}
		return newPair.m_newShape;
	}
}

//
//	This will try to scale the given shape by the given amount.

hknpShape* hknpShapeScalingUtil::scaleShape(const hknpShape* shape, hkVector4Parameter vScale, hkArray<ShapePair>* doneShapes)
{
	hkMatrix4 worldFromShape;
	worldFromShape.setIdentity();

	if ( doneShapes )
	{
		return hknpShapeScalingUtilImpl::scaleShape(worldFromShape, shape, vScale, *doneShapes);
	}

	// Make a temp one
	{

		hkArray<ShapePair> shapeCache;
		return hknpShapeScalingUtilImpl::scaleShape(worldFromShape, shape, vScale, shapeCache);
	}
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

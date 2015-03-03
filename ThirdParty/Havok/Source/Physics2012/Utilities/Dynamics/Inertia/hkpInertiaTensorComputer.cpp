/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Dynamics/hkpDynamics.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesConnectivityUtil.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Deprecated/MultiSphere/hkpMultiSphereShape.h>
#include <Physics2012/Collide/Shape/Misc/Bv/hkpBvShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>

// Compiler bug in SNC for TC 310
#if defined(HK_PLATFORM_PS3_PPU) && defined(HK_COMPILER_SNC) && (__SN_VER__ == 31001)
_Pragma("control %push O=1")
#endif


HK_FORCE_INLINE static void scaleMassProperties(hkSimdRealParameter scale, hkMassProperties& massProperties)
{
	// We change the MASS values only
	hkSimdReal m; 
	m.load<1>(&massProperties.m_mass);
	m.mul(scale);
	massProperties.m_inertiaTensor.mul( scale );
	m.store<1>(&massProperties.m_mass);
}

void HK_CALL hkpInertiaTensorComputer::setShapeVolumeMassProperties(const hkpShape* shape, hkReal mass, hkpRigidBodyCinfo& bodyInfo)
{
	HK_ASSERT2(0x69e12af9, shape != HK_NULL, "shape is NULL in setShapeVolumeMassProperties()!");

	hkMassProperties massProperties;
	computeShapeVolumeMassProperties( shape, mass, massProperties );

		// Extract the relevant info
	bodyInfo.m_mass			 = massProperties.m_mass;
	bodyInfo.m_centerOfMass  = massProperties.m_centerOfMass;
	bodyInfo.m_inertiaTensor = massProperties.m_inertiaTensor;
}

void HK_CALL hkpInertiaTensorComputer::setMassProperties(const hkMassProperties& massProperties, hkpRigidBodyCinfo& bodyInfo)
{
	// Extract the relevant info
	bodyInfo.m_mass			 = massProperties.m_mass;
	bodyInfo.m_centerOfMass  = massProperties.m_centerOfMass;
	bodyInfo.m_inertiaTensor = massProperties.m_inertiaTensor;
}

void HK_CALL hkpInertiaTensorComputer::setAndScaleToMass(const hkMassProperties& props, hkSimdRealParameter mass, hkpRigidBodyCinfo& bodyInfo)
{
	HK_ASSERT2( 0xf0ed3de1, props.m_mass > hkReal(0), "You need to pass in a valid mass" );

	hkSimdReal m; m.load<1>(&props.m_mass);
	hkSimdReal f; f.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(mass, m);

	// Extract the relevant info
	mass.store<1>(&bodyInfo.m_mass);
	bodyInfo.m_centerOfMass  = props.m_centerOfMass;
	bodyInfo.m_inertiaTensor = props.m_inertiaTensor;
	bodyInfo.m_inertiaTensor.mul(f);
}

void HK_CALL hkpInertiaTensorComputer::setAndScaleToDensity(const hkMassProperties& props, hkSimdRealParameter density, hkpRigidBodyCinfo& bodyInfo)
{
	hkSimdReal newMass; 
	newMass.load<1>(&props.m_volume); 
	newMass.mul(density);
	setAndScaleToMass( props, newMass, bodyInfo );
}



void HK_CALL hkpInertiaTensorComputer::clipInertia( hkReal maxInertiaRatio, hkpRigidBodyCinfo& bodyInfo)
{
	hkVector4 diag; hkMatrix3Util::_getDiagonal(bodyInfo.m_inertiaTensor, diag);
	hkSimdReal maxInertia = diag.horizontalMax<3>();

#if 0 // Disable warning.
	hkSimdReal minInertia = diag.horizontalMin<3>();

	hkSimdReal actualRatio = maxInertia/minInertia;

	if(actualRatio/maxInertia > hkSimdReal::getConstant<HK_QUADREAL_2>())
	{
		HK_WARN_ALWAYS(0x7761D715, "Inertia ratio "<<actualRatio.getReal()<<" is very large compared to requested ratio ("<<maxInertiaRatio<<"), simulation artifacts may occurs.");
	}	
#endif

	maxInertia.div<HK_ACC_FULL,HK_DIV_IGNORE>(hkSimdReal::fromFloat(maxInertiaRatio));

	hkVector4 maxInertiaV; maxInertiaV.setAll(maxInertia);
	hkVector4 diagMax; diagMax.setMax(diag, maxInertiaV);

	hkMatrix3Util::_setDiagonalOnly(diagMax, bodyInfo.m_inertiaTensor);
}

static void HK_CALL computeRecursiveShapeVolumeMassProperties(const hkcdShape* shape, const hkTransform& transform, hkReal minTriangleThickness, hkMassProperties& massPropertiesOut);
static void HK_CALL transformAndCombineVolumeMassProperties(const hkQsTransform& transform, hkMassProperties& newMassProperties, hkMassProperties& massPropertiesOut);

void HK_CALL hkpInertiaTensorComputer::computeShapeVolumeMassProperties(const hkcdShape* shape, hkReal mass, hkMassProperties &result)
{
	HK_ASSERT2(0x4eaa489f, shape != HK_NULL, "shape is NULL in computeShapeVolumeMassProperties()!");
 
	hkMassElement element;

		// Default constructor already initializes these, but this makes
		// it explicit as to what the "uncomputed" values are.
	element.m_properties.m_centerOfMass.setZero();
	element.m_properties.m_inertiaTensor.setZero();
	element.m_properties.m_mass = hkReal(0);
	element.m_properties.m_volume = hkReal(0);
	element.m_transform.setIdentity();

	hkReal minTriangleThickness = hkReal(0);
	computeRecursiveShapeVolumeMassProperties(shape, element.m_transform, minTriangleThickness, element.m_properties);

	if(element.m_properties.m_volume == hkReal(0))
	{
		HK_WARN(0x726807aa, "A shape with zero volume processed. Assuming minimum trinagle thickness of 0.01 and recalculating!");

		minTriangleThickness = hkReal(0.01f);
		computeRecursiveShapeVolumeMassProperties(shape, element.m_transform, minTriangleThickness, element.m_properties);

		if(element.m_properties.m_volume == hkReal(0))
		{
			HK_WARN(0x72680769, "Cannot call computeShapeVolumeMassProperties on a shape with no volume!");
			element.m_properties.m_centerOfMass.setZero();
			element.m_properties.m_inertiaTensor.setZero();
			element.m_properties.m_mass = hkReal(0);
			element.m_properties.m_volume = hkReal(0);
			return;
		}
	}

	// The above call assumes density = 1.0, so we must rescale using mass and volume.
	{
		hkSimdReal m; m.load<1>(&mass);
		hkSimdReal v; v.load<1>(&element.m_properties.m_volume);
		m.div<HK_ACC_FULL,HK_DIV_IGNORE>(v); // vol==0 checked above
		scaleMassProperties(m, element.m_properties);
	}

	result = element.m_properties;

}

// Compute mass properties of a shape with unscaled transform.
static void HK_CALL computeRecursiveShapeVolumeMassProperties(const hkcdShape* shape, const hkTransform& transform, hkReal minTriangleThickness, hkMassProperties& massPropertiesOut)
{
	hkMassProperties newMassProperties;

		// Here's the deal. We don't know what the final mass distribution should be, so we ASSUME uniform density
		// which means we can compute massProperties with mass=1, then scale to get values for DENSITY=1 using the volume
		// computed. This means we can ONLY work on shapes which have a non-zero volume.

		// If we can actually COMPUTE the mass properties given the shape type, we do so, and combine (below the
		// switch statement), otherwise we recurse ( eg. hkpTransformShape, hkpListShape etc.)
	HK_ASSERT2(0x3639fff5,  shape != HK_NULL, "Error: shape is NULL!");
	
	HK_ON_DEBUG(hkResult result = HK_FAILURE;)

	switch (shape->getType())
	{
		
		case hkcdShapeType::SPHERE:
		{
			const hkpSphereShape* sphereShape = static_cast<const hkpSphereShape*>(shape);
			HK_ON_DEBUG(result =) hkpInertiaTensorComputer::computeSphereVolumeMassProperties(sphereShape->getRadius(), hkReal(1), newMassProperties);
			break;
		}
		
		
		case hkcdShapeType::BOX:
		{
			const hkpBoxShape* boxShape = static_cast<const hkpBoxShape*>(shape);

			hkVector4 convexRadius;
			convexRadius.setZero(); convexRadius.setXYZ( boxShape->getRadius() );

			hkVector4 halfExtents = boxShape->getHalfExtents();
			halfExtents.add( convexRadius );

			HK_ON_DEBUG(result =) hkpInertiaTensorComputer::computeBoxVolumeMassProperties( halfExtents, hkReal(1), newMassProperties );
			break;
		}


		case hkcdShapeType::CONVEX_VERTICES:
		{
			const hkpConvexVerticesShape*	cvShape = static_cast<const hkpConvexVerticesShape*>( shape );
			/* Extract vertices									*/ 
			hkArray<hkVector4>	vertices;			
			cvShape->getOriginalVertices(vertices);
			/* Build mass properties of vertices + convexRadius	*/ 
			HK_ON_DEBUG(result =) hkpInertiaTensorComputer::computeConvexHullMassProperties( vertices, cvShape->getRadius(), newMassProperties );
			/* Checks										*/ 
			#if defined(HK_DEBUG)
				#if !defined (HK_PLATFORM_SPU)
				if(	!(	newMassProperties.m_volume>HK_REAL_EPSILON		&&
						newMassProperties.m_centerOfMass.isOk<3>()		&&
						newMassProperties.m_inertiaTensor.isOk()		&&
						newMassProperties.m_inertiaTensor.isSymmetric()))
				{				
					HK_REPORT("Invalid mass properties:");
					HK_REPORT("\tVolume: "<<newMassProperties.m_volume);
					hkStringBuf s;
					HK_REPORT("\tCOM: "<<hkVector4Util::toString3(newMassProperties.m_centerOfMass,s));
					HK_REPORT("\tIx0: "<<hkVector4Util::toString3(newMassProperties.m_inertiaTensor.getColumn(0),s));
					HK_REPORT("\tIx1: "<<hkVector4Util::toString3(newMassProperties.m_inertiaTensor.getColumn(1),s));
					HK_REPORT("\tIx2: "<<hkVector4Util::toString3(newMassProperties.m_inertiaTensor.getColumn(2),s));
				}
				#endif
			#endif
			break;
		}

		case hkcdShapeType::TRIANGLE:
		{
			const hkpTriangleShape* triangleShape = static_cast<const hkpTriangleShape*>(shape);
			const hkVector4& v0 = triangleShape->getVertex<0>();
			const hkVector4& v1 = triangleShape->getVertex<1>();
			const hkVector4& v2 = triangleShape->getVertex<2>();
			const hkReal surfaceThickness = hkMath::max2(triangleShape->getRadius(), minTriangleThickness); // use a minimum thickness of 1cm 
			HK_ON_DEBUG(result =) hkpInertiaTensorComputer::computeTriangleSurfaceMassProperties(v0, v1, v2, hkReal(1), surfaceThickness, newMassProperties);
			//if(triangleShape->getRadius() == 0.0f)
			//{
			//	HK_WARN(0x52f5f491, "Computing mass properties of triangle with thickness (radius) of 0. It has no volume, so mass cannot be distributed correctly in computeShapeVolumeMassProperties(), hence properties are ignored.\n");
			//}
			break;
		}


		case hkcdShapeType::MULTI_RAY:
		{
			// EMPTY - Rays have no volume!
			HK_WARN(0x57ca2117, "hkcdShapeType::MULTI_RAY has no volume, mass properties ignored.\n");
			return;
		}
		

		case hkcdShapeType::BV:
		{
			const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(shape);
		
			computeRecursiveShapeVolumeMassProperties(bvShape->getChildShape(), transform, minTriangleThickness, massPropertiesOut);

			// Return here, since the above call will have actually done the "addition" of the child's 
			// mass properties.
			return;
		}		
		case hkcdShapeType::CONVEX_TRANSLATE:
		{
			const hkpConvexTranslateShape* ts = static_cast<const hkpConvexTranslateShape*>( shape );

			hkTransform tst; tst.setIdentity();
			tst.setTranslation( ts->getTranslation() );

			// Must concatenate this transform on before recursing. 
			hkTransform t;			t.setMul(transform, tst);	
			const hkpShape* childShape = ts->getChildShape();
			computeRecursiveShapeVolumeMassProperties(childShape, t, minTriangleThickness, massPropertiesOut);

			// Return here, since the above call will have actually done the "addition" of the child's 
			// mass properties.
			return;
		}

		
			// Recurse here
		case hkcdShapeType::CONVEX_TRANSFORM:
		{
			const hkpConvexTransformShape* cts = static_cast<const hkpConvexTransformShape*>( shape );

			// First compute mass properties without transforms
			hkMassProperties childMassProperties;
			computeRecursiveShapeVolumeMassProperties(cts->getChildShape(), hkTransform::getIdentity(), minTriangleThickness, childMassProperties);

			// Now apply the concatenated transform before returning (first scale then rotation and translation)
			hkQsTransform parentTransform; parentTransform.setFromTransformNoScale(transform);
			hkQsTransform totalTransform; totalTransform.setMul(parentTransform, cts->getQsTransform());
			transformAndCombineVolumeMassProperties(totalTransform, childMassProperties, massPropertiesOut);

			// Return here, since the above call will have actually done the "addition" of the child's 
			// mass properties.
			return;
		}
		case hkcdShapeType::TRANSFORM:
		{
			const hkpTransformShape* ts = static_cast<const hkpTransformShape*>( shape );

			// Must concatenate this transform on before recursing. 
			hkTransform t;
			t.setMul(transform, ts->getTransform());	
			const hkpShape* childShape = ts->getChildShape();
			computeRecursiveShapeVolumeMassProperties(childShape, t, minTriangleThickness, massPropertiesOut);

			// Return here, since the above call will have actually done the "addition" of the child's 
			// mass properties.
			return;
		}

		case hkcdShapeType::MULTI_SPHERE:
		{
			const hkpMultiSphereShape* multiSphere = static_cast<const hkpMultiSphereShape*>(shape);	

			for (int i = 0; i < multiSphere->getNumSpheres(); i++ )
			{
				hkTransform t = transform;
				const hkVector4& sphere = multiSphere->getSpheres()[i];
				t.getTranslation()._setTransformedPos( t, sphere );
				
				hkpSphereShape sp( sphere(3) );
				computeRecursiveShapeVolumeMassProperties( &sp, t, minTriangleThickness, massPropertiesOut);
			}
			return;
		}

		case hkcdShapeType::TRIANGLE_COLLECTION: // [HVK-1821] concave mesh shapes. Should warn perhaps though.
		case hkcdShapeType::CONVEX_LIST:
		case hkcdShapeType::LIST:
		case hkcdShapeType::COLLECTION:
		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
		case hkcdShapeType::BV_TREE:
		case hkcdShapeType::MOPP:
		case hkcdShapeType::EXTENDED_MESH:
		case hkcdShapeType::STATIC_COMPOUND:
		case hkcdShapeType::BV_COMPRESSED_MESH:
		case hkcdShapeType::COMPRESSED_MESH:
		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE:
		{			
			const hkpShapeContainer* shapeContainer = static_cast<const hkpShape*>(shape)->getContainer();			
			
			// Loop through all children
			{
				hkpShapeBuffer buffer;
				for (	hkpShapeKey key = shapeContainer->getFirstKey();
						key != HK_INVALID_SHAPE_KEY;
						key = shapeContainer->getNextKey( key ))
				{
					const hkpShape* childShape = shapeContainer->getChildShape( key, buffer );
					if (childShape != HK_NULL)
					{
						computeRecursiveShapeVolumeMassProperties(childShape, transform, minTriangleThickness, massPropertiesOut);	
					}
					
				}
			}
			// Return here, since the above call will have actually done the "addition" of the child's 
			// mass properties.
			return;
		}

						

		case hkcdShapeType::PLANE:
		{
			// TODO - Plane shape not yet fully implemented.
			HK_ASSERT2(0x3bfb9e62,  0, "Cannot compute mass properties of hkcdShapeType::PLANE." );
			break;
		}


		case hkcdShapeType::CAPSULE:
		{
			const hkpCapsuleShape* capsuleShape = static_cast<const hkpCapsuleShape*>(shape);
			HK_ON_DEBUG(result =) hkpInertiaTensorComputer::computeCapsuleVolumeMassProperties(capsuleShape->getVertex<0>(), capsuleShape->getVertex<1>(), capsuleShape->getRadius(), hkReal(1), newMassProperties);
			break;
		}

		case hkcdShapeType::CYLINDER:
		{
			const hkpCylinderShape* cylinderShape = static_cast<const hkpCylinderShape*>(shape);

			const hkSimdReal convexRadius = hkSimdReal::fromFloat(cylinderShape->getRadius());
			hkVector4 vertex0 = cylinderShape->getVertex<0>();
			hkVector4 vertex1 = cylinderShape->getVertex<1>();
			
			hkVector4 direction;
			direction.setSub( vertex1, vertex0 );
			direction.normalize<3>();

			vertex0.subMul( convexRadius, direction );
			vertex1.addMul( convexRadius, direction );

			const hkReal cylinderRadius = cylinderShape->getCylinderRadius() + cylinderShape->getRadius();
			
			HK_ON_DEBUG(result =) hkpInertiaTensorComputer::computeCylinderVolumeMassProperties( vertex0, vertex1, cylinderRadius, hkReal(1), newMassProperties );
			
			break;
		}
		
		default:
		{
			HK_WARN(0x7d6a631a, "Unknown shape type to compute mass properties of. Have you implemented a user shape but not yet implemented mass property calculation in hkpInertiaTensorComputer::computeRecursiveShapeVolumeMassProperties()?\n" );
			return;
		}
		

	}

	HK_ASSERT2(0x900fb356, result == HK_SUCCESS, "computeRecursiveShapeVolumeMassProperties failed");

	// If volume of shape is zero, we cannot distribute the mass correctly, so we ignore this shape's contribution.
	if(newMassProperties.m_volume == hkReal(0))
	{
		return;
	}

	// First rescale the mass properties so density = 1.0f
	hkSimdReal m; m.load<1>(&newMassProperties.m_volume);
	scaleMassProperties(m, newMassProperties);

	
	{
		hkInplaceArrayAligned16<hkMassElement,2> allElements;

		// Even in the base case (where we just want to do massProperties = newMassProperties),
		// we can still use the call to combineMassProperties since the base massProperties will be
		// "zero", and will combine correctly with newMassProperties. cf. combineMassProperties()

		hkMassElement& currentMassElement = allElements.expandOne();
		currentMassElement.m_properties = massPropertiesOut;
		currentMassElement.m_transform.setIdentity();

		hkMassElement& newMassElement = allElements.expandOne();
		newMassElement.m_properties = newMassProperties;
		newMassElement.m_transform = transform;

		HK_ON_DEBUG(hkResult resultCombine =) hkpInertiaTensorComputer::combineMassProperties(allElements, massPropertiesOut);
		HK_ASSERT2(0x905fb356, resultCombine == HK_SUCCESS, "combineMassProperties failed");
	}	

}


void scaleVolumeMassProperties(const hkVector4& scale, hkMassProperties& massProperties)
{
	hkReal scaleX = scale(0);
	hkReal scaleY = scale(1);
	hkReal scaleZ = scale(2);
	hkReal xSqrDiv2 = ( (scaleX*scaleX) / 2 );
	hkReal ySqrDiv2 = ( (scaleY*scaleY) / 2 );
	hkReal zSqrDiv2 = ( (scaleZ*scaleZ) / 2 );

	hkMatrix3 newIner;
	hkMatrix3 currIner = massProperties.m_inertiaTensor;
	newIner(0,0) = (ySqrDiv2 * (currIner(0,0) + currIner(2,2) - currIner(1,1) ) ) + ( zSqrDiv2 * (currIner(0,0) + currIner(1,1) - currIner(2,2) ) );
	newIner(1,1) = (xSqrDiv2 * (currIner(1,1) + currIner(2,2) - currIner(0,0) ) ) + ( zSqrDiv2 * (currIner(0,0) + currIner(1,1) - currIner(2,2) ) );
	newIner(2,2) = (xSqrDiv2 * (currIner(1,1) + currIner(2,2) - currIner(0,0) ) ) + ( ySqrDiv2 * (currIner(0,0) + currIner(2,2) - currIner(1,1) ) );
	newIner(0,1) = currIner(0,1) * scaleX * scaleY;
	newIner(1,0) = newIner(0,1);
	newIner(0,2) = currIner(0,2) * scaleX * scaleZ;
	newIner(2,0) = newIner(0,2);
	newIner(1,2) = currIner(1,2) * scaleY * scaleZ;
	newIner(2,1) = newIner(1,2);

	massProperties.m_volume *= (scaleX * scaleY * scaleZ);
	massProperties.m_centerOfMass.mul(scale);
	massProperties.m_inertiaTensor = newIner;
}

static void HK_CALL transformAndCombineVolumeMassProperties(const hkQsTransform& transform, hkMassProperties& newMassProperties, hkMassProperties& massPropertiesOut)
{
	// Note: for child shapes that are hkpConvexVerticesShapes:
	// This function will calculate a different IT to the equivalent hkpConvexVerticesShape 
	// which contains the transformed versions of the child hkpConvexVerticesShape's vertices.
	// This is due to approximations made in computeConvexHullMassProperties to account for the convex radius

	if(newMassProperties.m_volume == hkReal(0))
	{
		return;
	}

	// test if the convex transform shape has identity scaling
	hkSimdReal eps; eps.setFromFloat(1e-3f);
	hkVector4 scale = transform.m_scale;
	hkBool unitScale = scale.allEqual<3>( hkVector4::getConstant<HK_QUADREAL_1>(), eps );
	if( !unitScale )
	{
		// scale (volume, COM & IT) before rotation and translation
		scaleVolumeMassProperties(scale, newMassProperties);
	}

	// Rescale the mass properties so density = 1.0f
	hkReal ratio = newMassProperties.m_volume / newMassProperties.m_mass;
	hkSimdReal m; m.load<1>(&ratio);
	scaleMassProperties(m, newMassProperties);

	{
		hkInplaceArrayAligned16<hkMassElement,2> allElements;

		// Even in the base case (where we just want to do massProperties = newMassProperties),
		// we can still use the call to combineMassProperties since the base massProperties will be
		// "zero", and will combine correctly with newMassProperties. cf. combineMassProperties()

		hkMassElement& currentMassElement = allElements.expandOne();
		currentMassElement.m_properties = massPropertiesOut;
		currentMassElement.m_transform.setIdentity();

		hkMassElement& newMassElement = allElements.expandOne();
		newMassElement.m_properties = newMassProperties;
		hkTransform trans; trans.setTranslation(transform.getTranslation()); trans.setRotation(transform.getRotation());
		newMassElement.m_transform = trans;

		HK_ON_DEBUG(hkResult resultCombine =) hkpInertiaTensorComputer::combineMassProperties(allElements, massPropertiesOut);
		HK_ASSERT2(0x905fb356, resultCombine == HK_SUCCESS, "combineMassProperties failed");
	}
}


static hkSimdReal HK_CALL hkInertiaTensorComputer_optimizeInertiasOfConstraintTreeInt(
			hkArray<const hkpConstraintInstance*>& constraints, 
			hkpRigidBody* body, hkSimdRealParameter inertiaFactorHint )
{

	// find all children
	hkInplaceArray<const hkpConstraintInstance*,16> children;
	{
		for (int i = constraints.getSize()-1; i>=0;i-- )
		{
			const hkpConstraintInstance* ci = constraints[i];
			hkpRigidBody* otherBody = HK_NULL;
			if (  ci->getRigidBodyA() == body )
			{
				otherBody = ci->getRigidBodyB();
			}
			else if (  ci->getRigidBodyB() == body )
			{
				otherBody = ci->getRigidBodyA();
			}
			if ( otherBody )
			{
				constraints.removeAt(i);
				children.pushBack( ci );
			}
		}
	}

	// recurse all children
	hkSimdReal childInertiaSum; childInertiaSum.setZero();
	hkSimdReal maxChild; maxChild.setZero();
	{
		for (int i = children.getSize()-1; i>=0;i-- )
		{
			const hkpConstraintInstance* ci = children[i];
			hkpRigidBody* otherBody = (  ci->getRigidBodyA() == body ) ? ci->getRigidBodyB() : ci->getRigidBodyA();
			const hkSimdReal childInertia = hkInertiaTensorComputer_optimizeInertiasOfConstraintTreeInt( constraints, otherBody, inertiaFactorHint );
			childInertiaSum.add(childInertia);
			maxChild.setMax( childInertia, maxChild );
		}
	}

	if ( body->isFixedOrKeyframed() )
	{
		return hkSimdReal_0;
	}

	hkMatrix3 inertia;
	body->getInertiaLocal( inertia );
	hkVector4 diag; hkMatrix3Util::_getDiagonal(inertia, diag);
	const hkSimdReal maxD = diag.horizontalMax<3>();

	// do not touch leaf inertias
	if ( childInertiaSum.isEqualZero() )
	{
		return maxD;
	}

	// modify inertia
	hkSimdReal minI; 
	minI.setMin( childInertiaSum, maxD * inertiaFactorHint );
	minI.setMax( minI, maxChild );
	{
		hkVector4 minIV; minIV.setAll(minI);
		hkVector4 maxDiag; maxDiag.setMax(diag, minIV);
		hkMatrix3Util::_setDiagonalOnly(maxDiag, inertia);

		body->setInertiaLocal( inertia );
	}

	hkSimdReal maxI; 
	maxI.setMin( maxD * inertiaFactorHint, maxD + childInertiaSum );
	maxI.setMax( maxI, minI );

	return maxI;
}

void hkpInertiaTensorComputer::optimizeInertiasOfConstraintTree( hkpConstraintInstance*const* constraints, int numConstraints, hkpRigidBody* rootBody, hkReal inertiaFactorHint  )
{
	hkLocalArray<const hkpConstraintInstance*> constr(numConstraints);
	for (int i = 0; i < numConstraints; i++ )
	{
		constr.pushBackUnchecked( constraints[i] );
	}
	/*hkSimdReal overallMax =*/ hkInertiaTensorComputer_optimizeInertiasOfConstraintTreeInt( constr, rootBody, hkSimdReal::fromFloat(inertiaFactorHint) );
	//HK_REPORT("optimizeInertiasOfConstraintTree max = "<<overallMax.getReal());
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

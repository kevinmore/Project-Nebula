/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastCylinder.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>


#define HK_USING_APPROXIMATE_FLOAT_TO_INT_CONVERSION
#define HK_CONVERSION_TOLERANCE 0.05f

#if !defined(HK_PLATFORM_SPU)

hkpCylinderShape::hkpCylinderShape( const hkVector4& vertexA, const hkVector4& vertexB, hkReal cylinderRadius, hkReal paddingRadius )
: hkpConvexShape(HKCD_SHAPE_TYPE_FROM_CLASS(hkpCylinderShape), paddingRadius)
{
	initRoundUpThreshold();

	m_vertexA = vertexA;
	m_vertexB = vertexB;

#ifdef HK_DEBUG
	hkVector4 diff; diff.setSub( vertexB, vertexA );
	HK_ASSERT2( 0xf010345, diff.length<3>().getReal() != 0.0f, "You are not allowed to create a cylinder with identical vertices");
#endif

	// Set the actual radius of the cylinder
	setCylinderRadius( cylinderRadius );
	presetPerpendicularVector();

	m_cylBaseRadiusFactorForHeightFieldCollisions = 0.8f;
}

hkpCylinderShape::hkpCylinderShape( hkFinishLoadedObjectFlag flag ) : hkpConvexShape(flag)
{
	if (flag.m_finishing)
	{
		initRoundUpThreshold();
	}
	setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpCylinderShape));
}

#endif

static void assertRoundUpThreshold(hkReal intRoundUpThreshold)
{
#define msg "hkpCylinderShape::s_intRoundUpThreshold is not set properly, most likely the FPU mode has been changed. Reset the mode or reinitialize the threshold again."
	HK_ASSERT2(0xad5674dd, hkMath::hkToIntFast(1 - intRoundUpThreshold) == 1, msg);
	HK_ASSERT2(0xad5674de, hkMath::hkToIntFast(1 - intRoundUpThreshold - 0.01f) == 0, msg);
#undef msg
}

hkReal hkpCylinderShape::s_virtualTesselationParameter    = (2.0f - 2.0f * HK_CONVERSION_TOLERANCE) / 0.707f;
hkReal hkpCylinderShape::s_virtualTesselationParameterInv = 1.0f / ((2.0f - 2.0f * HK_CONVERSION_TOLERANCE) / 0.707f);
hkReal hkpCylinderShape::s_intRoundUpThreshold = -1.0f;

void HK_CALL hkpCylinderShape::setNumberOfVirtualSideSegments(int numSegments)
{
	HK_ASSERT2(0xad1bb4e3, numSegments >= 8 && numSegments <= 8*16, "hkpCylinderShape::setNumberOfVirtualSideSegments only accepts values between 8 and 128.");
	if (numSegments%8 != 0 )
	{
		HK_WARN(0xad1bb4e3, "Number of cylinder side segments rounded down to a multiple of 8.");
	}
	const hkReal value = (hkReal(numSegments / 8) - 2.0f * HK_CONVERSION_TOLERANCE) / 0.707f;  // 0.707f == sin(pi/4)
	s_virtualTesselationParameter = value;
	s_virtualTesselationParameterInv = 1.0f / value;
}

void hkpCylinderShape::presetPerpendicularVector()
{
	hkVector4 unitAxis;
	unitAxis.setSub(m_vertexB, m_vertexA);
	unitAxis.normalize<3>();
	hkVector4Util::calculatePerpendicularVector( unitAxis, m_perpendicular1 );

	m_perpendicular1.normalize<3>();
	m_perpendicular2.setCross(unitAxis, m_perpendicular1);
	//m_perpendicular2 must be a unit vector
}

hkReal hkpCylinderShape::getCylinderRadius() const
{
	return m_cylRadius;
}

void hkpCylinderShape::setCylinderRadius(const hkReal radius)
{
	m_cylRadius = radius;

	// updates the radius of the spheres representing the cylinder
	m_vertexA(3) = ( radius + m_radius );
	m_vertexB(3) = ( radius + m_radius );
}

void hkpCylinderShape::decodeVertexId(hkpVertexId code, hkVector4& result) const
{
	hkBool32 baseA     = (code >> VERTEX_ID_ENCODING_IS_BASE_A_SHIFT) & 1;
	hkBool32 sinSign   = (code >> VERTEX_ID_ENCODING_SIN_SIGN_SHIFT) & 1;
	hkBool32 cosSign   = (code >> VERTEX_ID_ENCODING_COS_SIGN_SHIFT) & 1;
	hkBool32 sinLesser = (code >> VERTEX_ID_ENCODING_IS_SIN_LESSER_SHIFT) & 1;

	// get the last 12 bytes for the value
	hkSimdReal value     = hkSimdReal::fromFloat(hkReal(code & VERTEX_ID_ENCODING_VALUE_MASK));
	value.add(hkSimdReal_Inv2);
	value.mul(hkSimdReal::fromFloat(s_virtualTesselationParameterInv));
	hkSimdReal other = (hkSimdReal_1 - value*value).sqrt();

	hkSimdReal sin, cos;
	if (sinLesser)
	{
		sin = value;
		cos = other;
	}
	else
	{
		cos = value;
		sin = other;
	}

	// calc both cos & sin then
	// sin = sinLesser * value + (1 - sinLesser) hkMath::sqrt(1-value*value)

	if (!sinSign)
	{
		sin = -sin;
	}
	if (!cosSign)
	{
		cos = -cos;
	}
	//sin = (-1 + 2 * sinSign) * sin;
	//cos = (-1 + 2 * cosSign) * sin;

	hkVector4 radius;
	{
		hkVector4 tmp1, tmp2;
		tmp1.setMul(cos, m_perpendicular1);
		tmp2.setMul(sin, m_perpendicular2);
		radius.setAdd(tmp1, tmp2);
	}
	radius.mul(hkSimdReal::fromFloat(m_cylRadius));
	result.setAdd( getVertex(1-baseA) , radius );
}

void hkpCylinderShape::getSupportingVertex(hkVector4Parameter direction, hkcdVertex& supportingVertexOut) const
{
	// direction is already in "this" space, so:

	// this function returns a point on the cylinder and ignores hkConvesShape::m_radius

	//
	// Generate vertexId
	//

	hkVector4 axis; axis.setSub(m_vertexB, m_vertexA);

	hkSimdReal cos = m_perpendicular1.dot<3>(direction);
	hkSimdReal sin = m_perpendicular2.dot<3>(direction);

	const hkSimdReal len2 = sin * sin + cos * cos;
	if (len2 >= hkSimdReal_EpsSqrd )
	{
		const hkSimdReal invLen = len2.sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>();
		sin.mul(invLen);
		cos.mul(invLen);
	}
	else
	{
		sin.setZero();
		cos = hkSimdReal_1;
	}

	const int sinSign = sin.isGreaterEqualZero() ? 1 : 0;
	const int cosSign = cos.isGreaterEqualZero() ? 1 : 0;
	int sinLesser;
	hkSimdReal value;
	{
		hkSimdReal usin; usin.setAbs(sin);
		hkSimdReal ucos; ucos.setAbs(cos);

		const hkVector4Comparison sLessC = usin.lessEqual(ucos);
		sinLesser = sLessC.anyIsSet() ? 1 : 0;
		value.setSelect(sLessC, usin, ucos);
	}
	// remember on which base the point lies
	const int baseA = axis.dot<3>(direction).isLessEqualZero() ? 1 : 0;

	// encode that info
	int code = 0;
	//
	// Cylinder agent info -- now it is important to synch the  hkPredGskCylinderAgent3 if you change the encoding of the virtual vertices.
	//
	code += baseA     << VERTEX_ID_ENCODING_IS_BASE_A_SHIFT;
	code += sinSign   << VERTEX_ID_ENCODING_SIN_SIGN_SHIFT;
	code += cosSign   << VERTEX_ID_ENCODING_COS_SIGN_SHIFT;
	code += sinLesser << VERTEX_ID_ENCODING_IS_SIN_LESSER_SHIFT;
	HK_ASSERT( 0xf02dfad4, code < 0x10000);
	// got 12 bits to store the (non-negative) value (4096 values)
	// we actually wont use the upper ~1/3 of the range now...
	HK_ASSERT2(0x3bd0155e, value.getReal() >= 0 && value.getReal() < 0.708f, "Value used to encode support vertex for the cylinder is negative (and it cannot be).");
#ifndef HK_USING_APPROXIMATE_FLOAT_TO_INT_CONVERSION
	const int intValue = int( value.getReal() * s_virtualTesselationParameter);
#else
	assertRoundUpThreshold(s_intRoundUpThreshold);
	int intValue = hkMath::hkToIntFast( value.getReal() * s_virtualTesselationParameter - s_intRoundUpThreshold + HK_CONVERSION_TOLERANCE);
#endif
	HK_ASSERT2(0x3bd0155f, intValue >= 0 && hkReal(intValue) < s_virtualTesselationParameter, "Fast float-to-int conversion returned an invalid value (in cylinder code).");
	HK_ASSERT2(0x3bd0155e, (intValue & VERTEX_ID_ENCODING_VALUE_MASK) == intValue, "The vertexId's value being encoded doesn't fit. Possible cause -- hkpCylinderShape::s_virtualTesselationParameter too large (> 16).");
	code += intValue;

	// calculate the position of the encoded vertex -- that way you know exactly what value is in cache
	decodeVertexId( hkpVertexId(code), supportingVertexOut);

	supportingVertexOut.setInt24W(code);

}

void hkpCylinderShape::convertVertexIdsToVertices(const hkpVertexId* ids, int numIds, hkcdVertex* verticesOut) const
{
	for (int i = numIds-1; i>=0; --i, ++verticesOut, ++ids)
	{
		hkpVertexId id = ids[0];
		decodeVertexId(id, *verticesOut);
		verticesOut->setInt24W( id );
	}
}

void hkpCylinderShape::getCentre(hkVector4& centreOut) const
{
	centreOut.setAdd(m_vertexA, m_vertexB);
	centreOut.mul(hkSimdReal_Inv2);
}

#ifndef HK_PLATFORM_SPU

void hkpCylinderShape::getFirstVertex(hkVector4& v) const
{
	v = getVertex<1>();
}

#endif

const hkSphere* hkpCylinderShape::getCollisionSpheres(hkSphere* sphereBuffer) const
{
	const hkReal cylinder_radius = m_cylRadius;
	hkReal cylRadiusForBaseSpheres = cylinder_radius;
	hkReal largeSphereDisplacement = 0.0f;
	{
		hkVector4 symmetryAxis; symmetryAxis.setSub( getVertex<1>(), getVertex<0>() );
		hkReal heightSqr = symmetryAxis.lengthSquared<3>().getReal();
		if (heightSqr >= 2.0f * 2.0f * cylinder_radius * cylinder_radius)
		{
			// Adding large spheres for smooth rolling
			largeSphereDisplacement = cylinder_radius;
		    cylRadiusForBaseSpheres = cylinder_radius * m_cylBaseRadiusFactorForHeightFieldCollisions;
		}
	}


	hkVector4 diag1; diag1.setAdd(m_perpendicular1, m_perpendicular2);
	diag1.mul( hkSimdReal::fromFloat(1.0f / 1.4142135623730950488016887242097f * cylRadiusForBaseSpheres ));
	hkVector4 diag2; diag2.setSub(m_perpendicular1, m_perpendicular2);
	diag2.mul( hkSimdReal::fromFloat(1.0f / 1.4142135623730950488016887242097f * cylRadiusForBaseSpheres));
	hkVector4 perp1; perp1.setMul(hkSimdReal::fromFloat(cylRadiusForBaseSpheres), m_perpendicular1);
	hkVector4 perp2; perp2.setMul(hkSimdReal::fromFloat(cylRadiusForBaseSpheres), m_perpendicular2);

	perp1.zeroComponent<3>();
	perp2.zeroComponent<3>();
	diag1.zeroComponent<3>();
	diag2.zeroComponent<3>();

	const hkReal convex_radius = m_radius;
	for (int cap = 0; cap < 2; ++cap)
	{
		hkVector4* s = reinterpret_cast<hkVector4*>(sphereBuffer + 8 * cap);
		hkVector4 baseCenter = getVertex(cap);
		baseCenter(3) = convex_radius;

		s[0].setAdd(baseCenter, perp1);
		s[1].setAdd(baseCenter, diag1);
		s[2].setAdd(baseCenter, perp2);
		s[3].setSub(baseCenter, diag2);
		s[4].setSub(baseCenter, perp1);
		s[5].setSub(baseCenter, diag1);
		s[6].setSub(baseCenter, perp2);
		s[7].setAdd(baseCenter, diag2);
	}

	// When the cylinder is long enough, and we can squeeze a big sphere (with its radius equal to the cylinder radius) into it -- we do.
	{
		hkVector4& s0 = *reinterpret_cast<hkVector4*>(sphereBuffer + 16);
		hkVector4& s1 = *reinterpret_cast<hkVector4*>(sphereBuffer + 17);

		// put extra large/center spheres in
		hkVector4 symmetryAxisVersor; symmetryAxisVersor.setCross(m_perpendicular1, m_perpendicular2);
		HK_ON_DEBUG(hkVector4 symmetryAxis; symmetryAxis.setSub( getVertex<1>(), getVertex<0>() ) );
		HK_ASSERT2(0x708e02c3, symmetryAxisVersor.dot<3>(symmetryAxis).getReal() > 0, "Internal error: wrong axis direction.");

		s0.setAddMul( getVertex<0>(), symmetryAxisVersor, hkSimdReal::fromFloat( largeSphereDisplacement ));
		s1.setAddMul( getVertex<1>(), symmetryAxisVersor, hkSimdReal::fromFloat(-largeSphereDisplacement ));
		s0(3) = largeSphereDisplacement + convex_radius;
		s1(3) = largeSphereDisplacement + convex_radius;
	}

	return sphereBuffer;
}

void hkpCylinderShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	// Transform axis vertices
	hkVector4 vertices[2];
	hkVector4Util::transformPoints(localToWorld, getVertices(), 2, vertices);

	// Calculate the expansion E we have to apply to the aabb enclosing the axis A to account for the 
	// cylinder radius R:
	// A = V1 - V0
	// Ex = R * SQRT( (Ay^2 + Az^2) / (Ax^2 + Ay^2 + Az^2) )
	// Ey = R * SQRT( (Ax^2 + Az^2) / (Ax^2 + Ay^2 + Az^2) )
	// Ez = R * SQRT( (Ax^2 + Ay^2) / (Ax^2 + Ay^2 + Az^2) )
	hkVector4 expansion;
	{
		// Calculate some intermediate results
		hkVector4 axis; axis.setSub(vertices[1],vertices[0]);

		const hkSimdReal axisLen2 = axis.lengthSquared<3>();
		hkSimdReal invAxisLen2; invAxisLen2.setReciprocal(axisLen2); 

		hkVector4 axisYZX; axisYZX.setPermutation<hkVectorPermutation::YZXW>(axis);
		hkVector4 axisZXY; axisZXY.setPermutation<hkVectorPermutation::ZXYW>(axis);

		hkVector4 root;
		root.setMul(axisYZX, axisYZX);
		root.addMul(axisZXY, axisZXY);
		root.mul(invAxisLen2);
		root.zeroIfTrue(root.less(hkVector4::getConstant<HK_QUADREAL_EPS>()));
		expansion.setSqrt(root);
		expansion.mul(hkSimdReal::fromFloat(m_cylRadius));			
	}

	// Add the convex radius and tolerance parameters
	hkSimdReal convexRadius; convexRadius.setFromFloat(tolerance + m_radius);
	expansion.setAdd(expansion,convexRadius);

	// Apply the expansion
	out.m_min.setMin(vertices[0], vertices[1]);
	out.m_min.sub(expansion);
	out.m_max.setMax(vertices[0], vertices[1]);
	out.m_max.add(expansion);
}


hkBool hkpCylinderShape::castRay(const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& results) const
{
	HK_TIMER_BEGIN("rcCylinder", HK_NULL);

	hkcdRay ray;
	ray.setEndPoints( input.m_from, input.m_to );

	hkSimdReal cvxRadius;	cvxRadius.setFromFloat(m_radius);
	hkSimdReal cylRadius;	cylRadius.setFromFloat(m_cylRadius);
	const hkSimdReal radius = cvxRadius + cylRadius;
	
	// Grow cylinder by the convex radius
	hkVector4 vStart = m_vertexA;
	hkVector4 vEnd = m_vertexB;
	hkVector4 vAxis;
	vAxis.setSub(vEnd, vStart);
	vAxis.normalize<3>();
	vStart.subMul(cvxRadius, vAxis);
	vEnd.addMul(cvxRadius, vAxis);

	// Intersect
	hkSimdReal hitFraction;	hitFraction.setFromFloat(results.m_hitFraction);
	hkVector4 vN;
	hkBool32 hasHit = hkcdRayCastCylinder(ray, vStart, vEnd, radius, &hitFraction, &vN);
	if ( hasHit )
	{
		results.m_normal = vN;
		results.m_hitFraction = hitFraction.getReal(); // This is a parameterization along the ray
		results.setKey( HK_INVALID_SHAPE_KEY );
		HK_TIMER_END();
		return true;
	}

	HK_TIMER_END();
	return false;
}

// assert to verify the correctness of getVertices()
HK_COMPILE_TIME_ASSERT( sizeof(hkVector4) == sizeof(hkSphere) );

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

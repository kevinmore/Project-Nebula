/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h> // Precompiled Header
#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Base/Types/Geometry/hkStridedVertices.h>

// Compiler bug in SNC for TC 310
#if defined(HK_PLATFORM_PS3_PPU) && defined(HK_COMPILER_SNC) && (__SN_VER__ == 31001)
_Pragma("control %push O=1")
#endif

	// Set simple convex hull mass properties calculation method if accurate calculation is stripped
hkInertiaTensorComputer::ConvexHullMassPropertiesFunction hkInertiaTensorComputer::s_computeConvexHullMassPropertiesFunction;


	// This simply multiplies all elements of the Inertia Tensor, and the mass, by the given scale.
	// It is used by other hkInertiaTensorComputer methods to rescale mass properties calculated assuming a uniform density.
HK_FORCE_INLINE static void scaleMassProperties(hkSimdRealParameter scale, hkMassProperties& massProperties)
{
	// We change the MASS values only
	hkSimdReal m; 
	m.load<1>(&massProperties.m_mass);
	m.mul(scale);
	massProperties.m_inertiaTensor.mul( scale );
	m.store<1>(&massProperties.m_mass);
}


hkResult HK_CALL hkInertiaTensorComputer::computeSphereVolumeMassProperties( hkReal radius, hkReal mass, hkMassProperties& result)
{
	HK_ASSERT2(0x5e1910e9,  mass > hkReal(0), "Cannot calculate mass properties with zero mass or less." );
	hkSimdReal m; m.load<1>(&mass);
	if (m.isLessEqualZero())
	{
		return HK_FAILURE;
	}

	HK_ASSERT2(0x63e00cfb,  radius > hkReal(0), "Cannot calculate sphere mass properties with zero radius or less." );
	hkSimdReal r; r.load<1>(&radius);
	if (r.isLessEqualZero())
	{
		return HK_FAILURE;
	}

	{
		const hkSimdReal k = m * r * r * hkSimdReal::fromFloat(0.4f);
		hkMatrix3Util::_setDiagonal(k, result.m_inertiaTensor);
	}

	result.m_centerOfMass.setZero();
	(hkSimdReal_4PiOver3 * r * r * r).store<1>(&result.m_volume);
	result.m_mass = mass;

	return HK_SUCCESS;
}

		
hkResult HK_CALL hkInertiaTensorComputer::computeSphereSurfaceMassProperties( hkReal radius, hkReal mass, hkReal surfaceThickness, hkMassProperties& result)
{
	HK_ASSERT2(0x2030563b,  mass > hkReal(0), "Cannot calculate mass properties with zero mass or less." );
	hkSimdReal m; m.load<1>(&mass);
	if (m.isLessEqualZero())
	{
		return HK_FAILURE;
	}
		// Compute mass properties of 2 spheres and subtract.
	HK_ASSERT2(0x34617801,  radius > surfaceThickness, "Cannot calculate sphere surface mass properties when surfaceThickness greater than or equal to radius." );
	hkSimdReal r; r.load<1>(&radius);
	hkSimdReal s; s.load<1>(&surfaceThickness);
	if (r.isLessEqual(s))
	{
		return HK_FAILURE;
	}

	HK_ASSERT2(0x45dd597a,  surfaceThickness > hkReal(0), "Cannot calculate surface mass properties with zero or negative surfaceThickness." );
	if (s.isLessEqualZero())
	{
		return HK_FAILURE;
	}

	const hkSimdReal radiusBig = r;
	const hkSimdReal radiusSmall = r - s;
	
	const hkSimdReal t = hkSimdReal_4PiOver3;
	const hkSimdReal volBig = t * radiusBig * radiusBig * radiusBig;
	const hkSimdReal volSmall = t * radiusSmall * radiusSmall * radiusSmall;

	// Assume uniform density ( rho = 1.0f)
	const hkSimdReal massBig = /*1.0f **/ volBig;
	const hkSimdReal massSmall = /*1.0f **/ volSmall;

	hkMassProperties resultBig;
	hkMassProperties resultSmall;

	if ( HK_FAILURE == computeSphereVolumeMassProperties(radiusBig.getReal(), massBig.getReal(), resultBig) )
	{
		return HK_FAILURE;
	}
	if ( HK_FAILURE == computeSphereVolumeMassProperties(radiusSmall.getReal(), massSmall.getReal(), resultSmall) )
	{
		return HK_FAILURE;
	}

	result.m_centerOfMass.setZero();
	result.m_inertiaTensor = resultBig.m_inertiaTensor;
	result.m_inertiaTensor.sub(resultSmall.m_inertiaTensor);

	result.m_mass = resultBig.m_mass - resultSmall.m_mass;
	result.m_volume = resultBig.m_volume - resultSmall.m_volume;

	m.div<HK_ACC_FULL,HK_DIV_IGNORE>(volBig - volSmall);
	scaleMassProperties(m, result);

	return HK_SUCCESS;
}

hkResult HK_CALL hkInertiaTensorComputer::computeBoxSurfaceMassProperties(hkVector4Parameter halfExtents, hkReal mass, hkReal surfaceThickness, hkMassProperties& result)
{
	HK_ASSERT2(0x77fda9b2,  mass > hkReal(0), "Cannot calculate mass properties with zero mass or less." );
	hkSimdReal m; m.load<1>(&mass);
	if (m.isLessEqualZero())
	{
		return HK_FAILURE;
	}
		// Compute mass properties of 2 boxes and subtract.

	HK_ASSERT2(0x6eb97c45, (halfExtents(0) > surfaceThickness) && (halfExtents(1) > surfaceThickness) && (halfExtents(2) > surfaceThickness),
		"Cannot calculate box surface mass properties with surfaceThickness greater than or equal to min of half-extents" );
	hkVector4 surfaceThicknessV; surfaceThicknessV.setAll(surfaceThickness);
	if( ! halfExtents.greater(surfaceThicknessV).allAreSet<hkVector4ComparisonMask::MASK_XYZ>() )
	{
		return HK_FAILURE;
	}

	HK_ASSERT2(0x6f186ff2,  surfaceThickness > hkReal(0), "Cannot calculate surface mass properties with zero or negative surfaceThickness." );
	if (surfaceThicknessV.lessEqualZero().anyIsSet())
	{
		return HK_FAILURE;
	}

	hkVector4 halfExtentsSmall; halfExtentsSmall.setSub(halfExtents, surfaceThicknessV);

	const hkSimdReal volBig = hkSimdReal_8 * halfExtents.horizontalMul<3>();
	const hkSimdReal volSmall = hkSimdReal_8 * halfExtentsSmall.horizontalMul<3>();

	// Assume uniform density ( rho = 1.0f)
	const hkSimdReal massBig = /*1.0f **/ volBig;
	const hkSimdReal massSmall = /*1.0f **/ volSmall;

	hkMassProperties resultBig;
	hkMassProperties resultSmall;

	if ( HK_FAILURE == computeBoxVolumeMassProperties(halfExtents, massBig.getReal(), resultBig) )
	{
		return HK_FAILURE;
	}
	if ( HK_FAILURE == computeBoxVolumeMassProperties(halfExtentsSmall, massSmall.getReal(), resultSmall) )
	{
		return HK_FAILURE;
	}

	result.m_centerOfMass.setZero();
	result.m_inertiaTensor = resultBig.m_inertiaTensor;
	result.m_inertiaTensor.sub(resultSmall.m_inertiaTensor);

	result.m_mass = resultBig.m_mass - resultSmall.m_mass;
	result.m_volume = resultBig.m_volume - resultSmall.m_volume;

	m.div<HK_ACC_FULL,HK_DIV_IGNORE>(volBig - volSmall);
	scaleMassProperties(m, result);

	return HK_SUCCESS;
}

hkResult HK_CALL hkInertiaTensorComputer::computeTriangleSurfaceMassProperties(hkVector4Parameter v0, hkVector4Parameter v1, hkVector4Parameter v2, hkReal mass, hkReal surfaceThickness, hkMassProperties& result)
{
	HK_ASSERT2(0x4c9681be,  mass > hkReal(0), "Cannot calculate mass properties with zero mass or less." );
	hkSimdReal m; m.load<1>(&mass);
	if (m.isLessEqualZero())
	{
		return HK_FAILURE;
	}

	HK_ASSERT2(0x3ee58084,  surfaceThickness >= hkReal(0), "Cannot calculate surface mass properties with negative surfaceThickness." );
	hkSimdReal s; s.load<1>(&surfaceThickness);
	if (s.isLessZero())
	{
		return HK_FAILURE;
	}

	const hkSimdReal minTwiceArea = hkSimdReal::fromFloat(1e-5f);

	hkVector4 com;
	hkMatrix3 it;
	hkVector4 normal;
	{
		hkVector4 cb;
		hkVector4 ab;
		cb.setSub(v2,v1);
		ab.setSub(v0,v1);

		normal.setCross( cb , ab);
	}

	const hkSimdReal twiceArea = normal.length<3,HK_ACC_FULL,HK_SQRT_SET_ZERO>();
	// If thickness less than 1e-5, use exact formula for triangular lamina.
	if (s < minTwiceArea)
	{
		com.setAdd(v0,v1);
		com.add(v2);
		com.mul(hkSimdReal_Inv3);

		const hkSimdReal m_12 = m * hkSimdReal::fromFloat(1.0f/12.0f);
		const hkSimdReal nine = hkSimdReal::fromFloat(9.0f);

		hkVector4 diag;
		{
			hkVector4 com_2; com_2.setMul(com,com); com_2.mul(nine);
			hkVector4 v0_2; v0_2.setMul(v0,v0);
			hkVector4 v1_2; v1_2.setMul(v1,v1);
			hkVector4 v2_2; v2_2.setMul(v2,v2);

			diag.setAdd(com_2, v0_2);
			diag.add(v1_2);
			diag.add(v2_2);
			diag.mul(m_12);

			hkVector4 dYXX; dYXX.setPermutation<hkVectorPermutation::YXXY>(diag);
			hkVector4 dZZY; dZZY.setPermutation<hkVectorPermutation::ZZYY>(diag);
			diag.setAdd(dYXX,dZZY);

			diag.zeroComponent<3>(); // for clearing the columns 4th component
		}

		hkVector4 offDiag;
		{
			hkVector4 com_2;
			{
				hkVector4 comXXY; comXXY.setPermutation<hkVectorPermutation::XXYY>(com);
				hkVector4 comYZZ; comYZZ.setPermutation<hkVectorPermutation::YZZW>(com);
				com_2.setMul(comXXY,comYZZ); com_2.mul(nine);
			}
			hkVector4 v0_2;
			{
				hkVector4 v0XXY; v0XXY.setPermutation<hkVectorPermutation::XXYY>(v0);
				hkVector4 v0YZZ; v0YZZ.setPermutation<hkVectorPermutation::YZZW>(v0);
				v0_2.setMul(v0XXY,v0YZZ);
			}
			hkVector4 v1_2;
			{
				hkVector4 v1XXY; v1XXY.setPermutation<hkVectorPermutation::XXYY>(v1);
				hkVector4 v1YZZ; v1YZZ.setPermutation<hkVectorPermutation::YZZW>(v1);
				v1_2.setMul(v1XXY,v1YZZ);
			}
			hkVector4 v2_2;
			{
				hkVector4 v2XXY; v2XXY.setPermutation<hkVectorPermutation::XXYY>(v2);
				hkVector4 v2YZZ; v2YZZ.setPermutation<hkVectorPermutation::YZZW>(v2);
				v2_2.setMul(v2XXY,v2YZZ);
			}

			offDiag.setAdd(com_2, v0_2);
			offDiag.add(v1_2);
			offDiag.add(v2_2);
			offDiag.mul(-m_12);
		}

		{
			hkVector4 perm; perm.setPermutation<hkVectorPermutation::XXYY>(offDiag);
			it.getColumn<0>().setSelect<hkVector4ComparisonMask::MASK_XW>(diag,perm);
		}
		{
			it.getColumn<1>().setSelect<hkVector4ComparisonMask::MASK_YW>(diag,offDiag);
		}
		{
			hkVector4 perm; perm.setPermutation<hkVectorPermutation::YZZW>(offDiag);
			it.getColumn<2>().setSelect<hkVector4ComparisonMask::MASK_ZW>(diag,perm);
		}

		shiftInertiaToCom(com, m, it);
	}
	else
	{
			// If very small area triangle, approx by point mass. Strictly speaking we should approximate by
			// a "bar" of length = surfaceThickness, but we assume that surfaceThickness is small.
		if (twiceArea < minTwiceArea)
		{
			com.setAdd(v0,v1);
			com.add(v2);
			com.mul(hkSimdReal_Inv3);

			hkVector4 com_diag;
			{
				com_diag.setMul(com,com);

				hkVector4 dYXX; dYXX.setPermutation<hkVectorPermutation::YXXY>(com_diag);
				hkVector4 dZZY; dZZY.setPermutation<hkVectorPermutation::ZZYY>(com_diag);
				com_diag.setAdd(dYXX,dZZY);
				com_diag.mul(m);

				com_diag.zeroComponent<3>(); // for clearing the columns 4th component
			}

			hkVector4 com_off;
			{
				hkVector4 comXXY; comXXY.setPermutation<hkVectorPermutation::XXYY>(com);
				hkVector4 comYZZ; comYZZ.setPermutation<hkVectorPermutation::YZZW>(com);
				com_off.setMul(comXXY, comYZZ); 
				com_off.mul(-m);
			}

			{
				hkVector4 perm; perm.setPermutation<hkVectorPermutation::XXYY>(com_off);
				it.getColumn<0>().setSelect<hkVector4ComparisonMask::MASK_XW>(com_diag,perm);
			}
			{
				it.getColumn<1>().setSelect<hkVector4ComparisonMask::MASK_YW>(com_diag,com_off);
			}
			{
				hkVector4 perm; perm.setPermutation<hkVectorPermutation::YZZW>(com_off);
				it.getColumn<2>().setSelect<hkVector4ComparisonMask::MASK_ZW>(com_diag,perm);
			}
		}
		else
		{
			normal.normalize<3>();

			const hkSimdReal st = s * hkSimdReal_Inv2;

			hkMassProperties properties;		
			{
				hkGeometry triGeom;
				triGeom.m_vertices.setSize(6);
				triGeom.m_vertices[0].setAddMul(v0,normal,st);
				triGeom.m_vertices[1].setAddMul(v0,normal,-st);

				triGeom.m_vertices[2].setAddMul(v1,normal,st);
				triGeom.m_vertices[3].setAddMul(v1,normal,-st);

				triGeom.m_vertices[4].setAddMul(v2,normal,st);
				triGeom.m_vertices[5].setAddMul(v2,normal,-st);

				triGeom.m_triangles.setSize(8);
				triGeom.m_triangles[0].set(0, 2, 4);	// top
				triGeom.m_triangles[1].set(1, 5, 3);	// bottom

				triGeom.m_triangles[2].set(0, 3, 2);
				triGeom.m_triangles[3].set(0, 1, 3);

				triGeom.m_triangles[4].set(1, 0, 4);
				triGeom.m_triangles[5].set(1, 4, 5);

				triGeom.m_triangles[6].set(2, 5, 4);
				triGeom.m_triangles[7].set(2, 3, 5);

				hkInertiaTensorComputer::computeGeometryVolumeMassProperties(&triGeom, mass, properties);
			}
			
			com = properties.m_centerOfMass;
			it = properties.m_inertiaTensor;
		}
	}

	result.m_mass = mass;
	result.m_inertiaTensor = it;
	result.m_centerOfMass = com;
	(twiceArea * hkSimdReal_Inv2 * s).store<1>(&result.m_volume);

	return HK_SUCCESS;

}


hkResult HK_CALL hkInertiaTensorComputer::computeBoxVolumeMassProperties(hkVector4Parameter halfExtents, hkReal mass, hkMassProperties &result)
{
	result.m_centerOfMass.setZero();
	result.m_mass = mass;

	hkVector4 diag; diag.setZero();
	hkResult res = computeBoxVolumeMassPropertiesDiagonalized(halfExtents, mass, diag, result.m_volume);
	hkMatrix3Util::_setDiagonal(diag, result.m_inertiaTensor);

	return res;
}


hkResult HK_CALL hkInertiaTensorComputer::computeBoxVolumeMassPropertiesDiagonalized(hkVector4Parameter halfExtents, hkReal mass, hkVector4 &inertiaDiagonal, hkReal& volume)
{
	HK_ASSERT2(0x33baaf42,  mass > hkReal(0), "Cannot calculate mass properties with zero mass or less." );

	hkSimdReal k; k.load<1>(&mass);
	if (k.isLessEqualZero())
	{
		return HK_FAILURE;
	}
	k.mul(hkSimdReal_Inv3);

	hkVector4 he2; he2.setMul(halfExtents,halfExtents);
	hkVector4 he2YXX; he2YXX.setPermutation<hkVectorPermutation::YXXY>(he2);
	hkVector4 he2ZZY; he2ZZY.setPermutation<hkVectorPermutation::ZZYY>(he2);
	inertiaDiagonal.setAdd(he2YXX, he2ZZY);
	inertiaDiagonal.mul(k);
	inertiaDiagonal.setW(hkSimdReal_1);

	(halfExtents.horizontalMul<3>() * hkSimdReal_8).store<1>(&volume);

	return HK_SUCCESS;
}

hkResult HK_CALL hkInertiaTensorComputer::computeVertexHullVolumeMassProperties(const hkReal* vertexIn, int striding, int numVertices, hkReal mass, hkMassProperties &result)
{
	HK_ASSERT2(0x24ed5952,  mass > hkReal(0), "Cannot calculate mass properties with zero mass or less." );
	hkSimdReal m; m.load<1>(&mass);
	if(m.isLessEqualZero())	
	{
		return HK_FAILURE;
	}
	if(numVertices<1)
	{
		return HK_FAILURE;
	}
	hkStridedVertices	sv;
	sv.m_numVertices	=	numVertices;
	sv.m_striding		=	striding;
	sv.m_vertices		=	vertexIn;
	hkResult success = computeConvexHullMassProperties(sv,hkReal(0),result);
	result.scaleToMass(m);
	return success;
}

hkResult HK_CALL hkInertiaTensorComputer::computeVertexCloudMassProperties(const hkReal* vertexIn, int striding, int numVertices, hkReal mass, hkMassProperties &result)
{
	
	HK_ASSERT2(0x60c3e8fb,  numVertices > 0, "Cannot calculate mass properties with zero vertices." );

	if(numVertices <= 0)
	{
		return HK_FAILURE;
	}
	hkSimdReal invNumV; 
	{
		hkSimdReal nV; nV.setFromInt32(numVertices);
		invNumV.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(nV);
	}

		// Extract vertices into an array
	hkArray<hkVector4> verts;
	verts.setSize(numVertices);
	const hkReal *ptr = vertexIn;
	int i;
	for(i = 0; i < numVertices; i++)
	{
		verts[i].load<3,HK_IO_NATIVE_ALIGNED>(ptr);
		verts[i].zeroComponent<3>(); // not used, but make sure we don't calc with NaNs
		ptr = hkAddByteOffsetConst( ptr, striding );
	}

	result.m_mass = mass;

	hkSimdReal pointMass; pointMass.load<1>(&mass); pointMass.mul(invNumV);

	// We can do this calculation all-in-one (computing the inertia tensor in another frame
	// and shifting it back to be around the center of mass), but it's more accurate to compute the
	// center of mass first
	result.m_centerOfMass.setZero();
	for(i = 0; i < numVertices; i++)
	{
		result.m_centerOfMass.add(verts[i]);	
	}
	result.m_centerOfMass.setMul(invNumV, result.m_centerOfMass);

	hkMatrix3 inertiaTensor; inertiaTensor.setZero();
	for(i = 0; i < numVertices; i++)
	{
		hkVector4 r;
		r.setSub(verts[i], result.m_centerOfMass);

		hkVector4 diag;
		{
			diag.setMul(r,r);

			hkVector4 dYXX; dYXX.setPermutation<hkVectorPermutation::YXXY>(diag);
			hkVector4 dZZY; dZZY.setPermutation<hkVectorPermutation::ZZYY>(diag);
			diag.setAdd(dYXX,dZZY);
			diag.mul(pointMass);

			diag.zeroComponent<3>(); // for clearing the columns 4th component
		}

		hkVector4 offDiag;
		{
			hkVector4 rXXY; rXXY.setPermutation<hkVectorPermutation::XXYY>(r);
			hkVector4 rYZZ; rYZZ.setPermutation<hkVectorPermutation::YZZW>(r);
			offDiag.setMul(rXXY,rYZZ); 
			offDiag.mul(-pointMass);
		}

		{
			hkVector4 perm; perm.setPermutation<hkVectorPermutation::XXYY>(offDiag);
			hkVector4 col; col.setSelect<hkVector4ComparisonMask::MASK_XW>(diag,perm);
			inertiaTensor.getColumn<0>().add(col);
		}
		{
			hkVector4 col; col.setSelect<hkVector4ComparisonMask::MASK_YW>(diag,offDiag);
			inertiaTensor.getColumn<1>().add(col);
		}
		{
			hkVector4 perm; perm.setPermutation<hkVectorPermutation::YZZW>(offDiag);
			hkVector4 col; col.setSelect<hkVector4ComparisonMask::MASK_ZW>(diag,perm);
			inertiaTensor.getColumn<2>().add(col);
		}
	}
	result.m_inertiaTensor = inertiaTensor;

	return HK_SUCCESS;	
}


hkResult HK_CALL hkInertiaTensorComputer::computeCapsuleVolumeMassProperties(hkVector4Parameter startAxis, hkVector4Parameter endAxis, hkReal radius, hkReal mass, hkMassProperties& result)
{
	HK_ASSERT2(0x1ab5bfed,  mass > hkReal(0), "Cannot calculate mass properties with zero mass or less." );
	hkSimdReal m; m.load<1>(&mass);
	if(m.isLessEqualZero())
	{
		return HK_FAILURE;
	}

	HK_ASSERT2(0x18f8cf41,  radius > hkReal(0), "Cannot calculate capsule mass properties with zero radius or less. You may edit hkInertiaTensorComputer::computeCapsuleVolumeProperties() to bypass this assert if you are sure this is the behaviour you desire." );
	hkSimdReal r; r.load<1>(&radius);
	if(r.isLessEqualZero())
	{
		return HK_FAILURE;
	}
	
	// First determine a transform from the capsule to "canonical space": (0,0,0) is the center of the capsule
	// and the axis lies along the Z-axis

	hkVector4 axis;
	axis.setSub(endAxis, startAxis);
	const hkSimdReal height = axis.length<3,HK_ACC_FULL,HK_SQRT_SET_ZERO>();
	hkTransform capsuleToLocal;
	if (height.isGreaterZero())
	{
		axis.normalize<3>();

		const hkSimdReal axisZExtent = axis.getComponent<2>();
		hkSimdReal absAxisExtent; absAxisExtent.setAbs(axisZExtent);
		if(absAxisExtent < hkSimdReal::fromFloat(1.0f - 1e-5f))
		{
			hkVector4 rotAxis;
			const hkVector4 canonicalZ = hkVector4::getConstant<HK_QUADREAL_0010>();
			rotAxis.setCross(canonicalZ, axis);
			rotAxis.normalize<3>();

			hkSimdReal rotAngle;
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
#if defined(HK_REAL_IS_DOUBLE)
			rotAngle.m_real = hkMath::twoAcos(axisZExtent.m_real);
#else
			rotAngle.m_real = hkMath::quadAcos(axisZExtent.m_real);
#endif
#else
			rotAngle.setFromFloat( hkMath::acos(axisZExtent.getReal()) );
#endif
			hkQuaternion q; q.setAxisAngle(rotAxis, rotAngle);
			capsuleToLocal.setRotation(q);
		}
		else
		{
			capsuleToLocal.setIdentity();	
		}

	}
	else
	{
		capsuleToLocal.setIdentity();	
	}


	const hkSimdReal half = hkSimdReal_Inv2;
	hkSimdReal zero; zero.setZero();
	
		// Now recenter

	{
		hkVector4 toCentre;
		toCentre.setAdd(startAxis, endAxis);
		toCentre.mul(half);
		capsuleToLocal.setTranslation(toCentre);
	}




	// Add together the mass properties of two hemispheres and a cylinder



	// Divide mass evenly amongst the caps and cylinder, based on volume
	const hkSimdReal volumeOfCaps = hkSimdReal_4PiOver3 * r * r * r;
	const hkSimdReal volumeOfCylinder = hkSimdReal_Pi * r * r * height;
	const hkSimdReal totalVolume = volumeOfCaps + volumeOfCylinder;
	HK_ASSERT(0x923842, totalVolume.isGreaterZero());

	hkSimdReal massOfCapsules = m * volumeOfCaps; massOfCapsules.div<HK_ACC_FULL,HK_DIV_IGNORE>(totalVolume);
	hkSimdReal massOfCylinder = m * volumeOfCylinder; massOfCylinder.div<HK_ACC_FULL,HK_DIV_IGNORE>(totalVolume);
	

	hkInplaceArray<hkMassElement,3> elements;

	
	// First do cylinder:
	// See Graphics Gems III p 142
	{
		hkMassElement cylinderElement;
		cylinderElement.m_transform = capsuleToLocal;

		const hkSimdReal quarter = hkSimdReal_Inv4;
		const hkSimdReal third = hkSimdReal_Inv3;

		const hkSimdReal f = ((r * r) + (height * height * third)) * quarter;
		hkVector4 diag; diag.set( f, f, r * r * half, quarter);
		hkMatrix3Util::_setDiagonal(diag, cylinderElement.m_properties.m_inertiaTensor);

			// Scale by mass
		cylinderElement.m_properties.m_inertiaTensor.mul(massOfCylinder);

		cylinderElement.m_properties.m_centerOfMass.setZero();

		volumeOfCylinder.store<1>(&cylinderElement.m_properties.m_volume);
		massOfCylinder.store<1>(&cylinderElement.m_properties.m_mass);

		elements.pushBack(cylinderElement);
	}
	

	
		// Top cap
	{
		hkMassElement capElement;
		capElement.m_transform = capsuleToLocal;
		hkVector4 pos; pos.set(zero, zero, (height * half), zero);
		pos._setRotatedDir(capsuleToLocal.getRotation(), pos);
		pos.add(capElement.m_transform.getTranslation());
		capElement.m_transform.setTranslation(pos);

		capElement.m_properties.m_centerOfMass.set(zero, zero, hkSimdReal::fromFloat(3.0f / 8.0f) * r, zero);
		// Now here's the tricky bit. TWO hemispheres make a sphere, hence "half a sphere IT" shifted by COM
		// should be the IT of a hemisphere, right? This only works because of the symmetry of the integrals x^2, y^2 and z^2
		// over the sphere.
		{
			hkMassProperties sphereResult;
			if (HK_FAILURE == computeSphereVolumeMassProperties(radius, massOfCapsules.getReal(), sphereResult))
			{
				HK_ASSERT2(0xf0021aa, false, "hkInertiaTensorComputer::computeCapsuleVolumeMassProperties failed");
				return HK_FAILURE;
			}

			capElement.m_properties.m_inertiaTensor = sphereResult.m_inertiaTensor;
			capElement.m_properties.m_inertiaTensor.mul(half);

			// Shift IT back
			shiftInertiaToCom(capElement.m_properties.m_centerOfMass, massOfCapsules * half, capElement.m_properties.m_inertiaTensor);
		}

		(volumeOfCaps * half).store<1>(&capElement.m_properties.m_volume);
		(massOfCapsules * half).store<1>(&capElement.m_properties.m_mass);

		elements.pushBack(capElement);
	}

	
		// Bottom cap
	{
		hkMassElement capElement;
		capElement.m_transform = capsuleToLocal;
		hkVector4 pos; pos.set(zero, zero, -(height * half), zero);
		pos._setRotatedDir(capsuleToLocal.getRotation(), pos);
		pos.add(capElement.m_transform.getTranslation());
		capElement.m_transform.setTranslation(pos);


		capElement.m_properties.m_centerOfMass.set(zero, zero, hkSimdReal::fromFloat(-3.0f / 8.0f) * r, zero);
		// Now here's the tricky bit. TWO hemispheres make a sphere, hence "half a sphere IT" shifted by COM
		// should be the IT of a hemisphere, right? This only works because of the symmetry of the integrals x^2, y^2 and z^2
		// over the sphere.
		{
			hkMassProperties sphereResult;
			if (HK_FAILURE == computeSphereVolumeMassProperties(radius, massOfCapsules.getReal(), sphereResult))
			{
				HK_ASSERT2(0xf0021aa, false, "hkInertiaTensorComputer::computeCapsuleVolumeMassProperties failed");
				return HK_FAILURE;
			}

			capElement.m_properties.m_inertiaTensor = sphereResult.m_inertiaTensor;
			capElement.m_properties.m_inertiaTensor.mul(half);

			// Shift IT back
			shiftInertiaToCom(capElement.m_properties.m_centerOfMass, massOfCapsules * half, capElement.m_properties.m_inertiaTensor);
		}

		(volumeOfCaps * half).store<1>(&capElement.m_properties.m_volume);
		(massOfCapsules * half).store<1>(&capElement.m_properties.m_mass);

		elements.pushBack(capElement);
	}
	
	return combineMassProperties(elements, result);
}

hkResult HK_CALL hkInertiaTensorComputer::computeCylinderVolumeMassProperties(hkVector4Parameter startAxis, hkVector4Parameter endAxis, hkReal radius, hkReal mass, hkMassProperties& result)
{
	// This is a simplified version of hkInertiaTensorCopmuter::computeCapsuleVolumeMassProperties

	HK_ASSERT2(0x1ab5bfed,  mass > hkReal(0), "Cannot calculate mass properties with zero mass or less." );
	hkSimdReal m; m.load<1>(&mass);
	if(m.isLessEqualZero())
	{
		return HK_FAILURE;
	}

	
	HK_ASSERT2(0x18f8cf41,  radius > hkReal(0), "Cannot calculate cylinder mass properties with zero radius or less. You may edit hkInertiaTensorComputer::computeCylinderVolumeProperties() to bypass this assert if you are sure this is the behaviour you desire." );
	hkSimdReal r; r.load<1>(&radius);
	if(r.isLessEqualZero())
	{
		return HK_FAILURE;
	}
	
	// First determine a transform from the capsule to "canonical space": (0,0,0) is the center of the cylinder
	// and the axis lies along the Z-axis

	hkVector4 axis;
	axis.setSub(endAxis, startAxis);
	const hkSimdReal height = axis.length<3,HK_ACC_FULL,HK_SQRT_SET_ZERO>();
	hkTransform capsuleToLocal;

	HK_ASSERT2(0x15938fa4, height.getReal() > hkReal(0), "Cannot calculate cylinder mass properties with zero height. You may edit hkInertiaTensorComputer::computeCylinderVolumeProperties() to bypass this assert if you are sure this is the behaviour you desire." );
	if (height.isLessEqualZero())
	{
		return HK_FAILURE;
	}

	{
		axis.normalize<3>();

		const hkSimdReal axisZExtent = axis.getComponent<2>();
		hkSimdReal absAxisExtent; absAxisExtent.setAbs(axisZExtent);
		if(absAxisExtent < hkSimdReal::fromFloat(1.0f - 1e-5f))
		{
			hkVector4 rotAxis;
			const hkVector4 canonicalZ = hkVector4::getConstant<HK_QUADREAL_0010>();
			rotAxis.setCross(canonicalZ, axis);
			rotAxis.normalize<3>();

			hkSimdReal rotAngle;
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
#if defined(HK_REAL_IS_DOUBLE)
			rotAngle.m_real = hkMath::twoAcos(axisZExtent.m_real);
#else
			rotAngle.m_real = hkMath::quadAcos(axisZExtent.m_real);
#endif
#else
			rotAngle.setFromFloat( hkMath::acos(axisZExtent.getReal()) );
#endif

			hkQuaternion q; q.setAxisAngle(rotAxis, rotAngle);
			capsuleToLocal.setRotation(q);
		}
		else
		{
			capsuleToLocal.setIdentity();	
		}
	}

	
		// Now recenter
	{
		hkVector4 toCentre;
		toCentre.setAdd(startAxis, endAxis);
		toCentre.mul(hkSimdReal_Inv2);
		capsuleToLocal.setTranslation(toCentre);
	}


	// Calculate the mass properties of a cylinder

	const hkSimdReal volumeOfCylinder = hkSimdReal_Pi * r * r * height;
	const hkSimdReal massOfCylinder = m;
	

	hkInplaceArray<hkMassElement,1> elements;
	
	// First do cylinder:
	// See Graphics Gems III p 142
	{
		hkMassElement& cylinderElement = elements.expandOne();
		cylinderElement.m_transform = capsuleToLocal;

		const hkSimdReal quarter = hkSimdReal_Inv4;
		const hkSimdReal third = hkSimdReal_Inv3;

		const hkSimdReal f = ((r * r) + (height * height * third)) * quarter;
		hkVector4 diag; diag.set( f, f, r * r * hkSimdReal_Inv2, quarter);
		hkMatrix3Util::_setDiagonal(diag, cylinderElement.m_properties.m_inertiaTensor);

			// Scale by mass
		cylinderElement.m_properties.m_inertiaTensor.mul(massOfCylinder);

		cylinderElement.m_properties.m_centerOfMass.setZero();

		volumeOfCylinder.store<1>(&cylinderElement.m_properties.m_volume);
		massOfCylinder.store<1>(&cylinderElement.m_properties.m_mass);
	}

	return combineMassProperties(elements, result);
}

hkResult HK_CALL hkInertiaTensorComputer::computeConvexHullMassProperties(const hkStridedVertices& vertices, hkReal radius, hkMassProperties& result)
{
	if( s_computeConvexHullMassPropertiesFunction )
	{
		return (*s_computeConvexHullMassPropertiesFunction)(vertices, radius, result);
	}
	else
	{
		return computeApproximateConvexHullMassProperties(vertices, radius, result);
	}
}


hkResult HK_CALL hkInertiaTensorComputer::computeGeometrySurfaceMassProperties(const hkGeometry* geom, hkReal surfaceThickness, hkBool distributeUniformly, hkReal mass, hkMassProperties &result)
{
	HK_ASSERT2(0x1f2bf574,  mass > hkReal(0), "Cannot calculate mass properties with zero mass or less." );
	hkSimdReal m; m.load<1>(&mass);
	if(m.isLessEqualZero())
	{
		return HK_FAILURE;
	}

	HK_ASSERT2(0x4071fc57,  surfaceThickness > hkReal(0), "Cannot calculate surface mass properties with zero or negative surfaceThickness." );
	hkSimdReal s; s.load<1>(&surfaceThickness);
	if(s.isLessEqualZero())
	{
		return HK_FAILURE;
	}
	


	result.m_inertiaTensor.setZero();
	result.m_centerOfMass.setZero();

	// First we need to find the mass of each triangle.
	// We'll say it's proportional to the area if distributeUniformly=true, otherwise we
	// give the same mass to each triangle, independent of its area. This means that "highly tessellated"
	// areas will have more mass, usually undesirable, but perhaps useful in some instances.
	int i;

	const hkSimdReal minTwiceArea = hkSimdReal::fromFloat(1e-5f);
	hkArray<hkSimdReal>	triArea;
	hkSimdReal totalArea;

	if (distributeUniformly)
	{
		triArea.setSize(geom->m_triangles.getSize());
		totalArea.setZero();
		for(i = 0; i < geom->m_triangles.getSize(); i++)
		{
			const hkGeometry::Triangle& tri = geom->m_triangles[i]; 
		
			hkVector4 normal;
			{
				const hkVector4& v0 = geom->m_vertices[tri.m_a];
				const hkVector4& v1 = geom->m_vertices[tri.m_b];
				const hkVector4& v2 = geom->m_vertices[tri.m_c];

				hkVector4 cb;
				hkVector4 ab;
				cb.setSub(v2,v1);
				ab.setSub(v0,v1);

				normal.setCross( cb , ab);
			}
			const hkSimdReal twiceArea = normal.length<3,HK_ACC_FULL,HK_SQRT_SET_ZERO>();

				// We have a minimum area allowed (so as to avoid breaking the convex hull code)
			triArea[i].setMax(minTwiceArea, twiceArea);
			triArea[i].mul(hkSimdReal_Inv2);

			totalArea.add(triArea[i]);
		}
		totalArea.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(totalArea);
	}
	else
	{
		// Assume every triangle has the same "area" eg. 1.0
		totalArea.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(hkSimdReal::fromInt32(geom->m_triangles.getSize()));
	}

	// For each triangle
	for(i = 0; i < geom->m_triangles.getSize(); i++)
	{
		const hkGeometry::Triangle& tri = geom->m_triangles[i]; 
	
		// OK, let's pick the triangle, expand it out, and use this as geom to pass to the volume integrator

		const hkVector4& v0 = geom->m_vertices[tri.m_a];
		const hkVector4& v1 = geom->m_vertices[tri.m_b];
		const hkVector4& v2 = geom->m_vertices[tri.m_c];

		hkSimdReal triMass = m * totalArea;
		if (distributeUniformly)
		{
			triMass.mul(triArea[i]);
		}

		hkMassProperties triMassProperties;
		if (HK_FAILURE == computeTriangleSurfaceMassProperties(v0, v1, v2, triMass.getReal(), surfaceThickness, triMassProperties))
		{
			HK_WARN_ONCE(0xfab7471, "hkInertiaTensorComputer::computeGeometrySurfaceMassProperties failed for one triangle");
		}
		else
		{
			const hkVector4& com = triMassProperties.m_centerOfMass;
			hkMatrix3& it = triMassProperties.m_inertiaTensor;

			result.m_centerOfMass.addMul(triMass, com);
			// To "add" a new inertia tensor, we must move it to a common space first
			shiftInertiaFromCom(com, triMass, it);
			result.m_inertiaTensor.add(it);			
		}
	}

	result.m_mass = mass;
	hkSimdReal invM; invM.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(m);
	result.m_centerOfMass.mul(invM);
	// Finally shift back to new center of mass
	shiftInertiaToCom(result.m_centerOfMass, m, result.m_inertiaTensor);

	return HK_SUCCESS;
}


hkResult HK_CALL hkInertiaTensorComputer::combineMassProperties(const hkArray<hkMassElement>& elements, hkMassProperties& result )
{
	// We'll have to move all ITs to a common space, and combine them. To determine the common space,
	// find the new center of mass.
	hkVector4 groupCenterOfMass;
	groupCenterOfMass.setZero();

	hkSimdReal groupMass; groupMass.setZero();
	hkSimdReal groupVolume; groupVolume.setZero();

	int i;
	for(i = 0; i < elements.getSize(); i++)
	{
		hkVector4 centerOfMassInCommon;
		centerOfMassInCommon._setTransformedPos(elements[i].m_transform, elements[i].m_properties.m_centerOfMass);
		hkSimdReal mass; mass.load<1>(&elements[i].m_properties.m_mass);
		hkSimdReal volume; volume.load<1>(&elements[i].m_properties.m_volume);
		groupCenterOfMass.addMul(mass, centerOfMassInCommon);
		
		groupMass.add(mass);
		groupVolume.add(volume);
	}


	if (groupMass.isLessEqualZero())
	{
		HK_ASSERT2(0x78b54666,  false, "Cannot calculate group mass properties with zero mass or less." );
		return HK_FAILURE;
	}

	hkSimdReal sgm; sgm.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>( groupMass );
	groupCenterOfMass.setMul(sgm, groupCenterOfMass);

	result.m_centerOfMass = groupCenterOfMass;
	groupMass.store<1>( &result.m_mass );
	groupVolume.store<1>( &result.m_volume );

	result.m_inertiaTensor.setZero();


	// We now have to "antishift" each IT from its center of mass, and rotate them to the common space
	// after which we can add them.
	for(i = 0; i < elements.getSize(); i++)
	{

		hkMatrix3 inertiaTensorInCommonSpace;
		inertiaTensorInCommonSpace = elements[i].m_properties.m_inertiaTensor;

				// First rotate.
		inertiaTensorInCommonSpace.changeBasis(elements[i].m_transform.getRotation());
				// Then shift.
		hkVector4 shift;
		shift._setTransformedPos(elements[i].m_transform, elements[i].m_properties.m_centerOfMass);
		shift.setSub(shift, groupCenterOfMass);
		hkSimdReal mass; mass.load<1>(&elements[i].m_properties.m_mass);
		shiftInertiaFromCom(shift, mass, inertiaTensorInCommonSpace);

		// Now can add to other inertia tensors
		result.m_inertiaTensor.add(inertiaTensorInCommonSpace);
	
	}
	return HK_SUCCESS;
}



void HK_CALL hkInertiaTensorComputer::shiftInertiaToCom(hkVector4Parameter shift, hkSimdRealParameter mass, hkMatrix3& inertia)
{
	hkVector4 diag;
	{
		diag.setMul(shift,shift);

		hkVector4 dYXX; dYXX.setPermutation<hkVectorPermutation::YXXY>(diag);
		hkVector4 dZZY; dZZY.setPermutation<hkVectorPermutation::ZZYY>(diag);
		diag.setAdd(dYXX,dZZY);
		diag.mul(-mass);

		diag.zeroComponent<3>(); // for clearing the columns 4th component
	}
	hkVector4 off_diag;
	{
		hkVector4 shiftYZX; shiftYZX.setPermutation<hkVectorPermutation::YZXW>(shift);
		off_diag.setMul(shift,shiftYZX); 
		off_diag.mul(mass);
	}

	{
		hkVector4 perm; perm.setPermutation<hkVectorPermutation::XXZZ>(off_diag);
		hkVector4 col; col.setSelect<hkVector4ComparisonMask::MASK_XW>(diag,perm);
		inertia.getColumn<0>().add(col);
	}
	{
		hkVector4 perm; perm.setPermutation<hkVectorPermutation::XXYY>(off_diag);
		hkVector4 col; col.setSelect<hkVector4ComparisonMask::MASK_YW>(diag,perm);
		inertia.getColumn<1>().add(col);
	}
	{
		hkVector4 perm; perm.setPermutation<hkVectorPermutation::ZYZW>(off_diag);
		hkVector4 col; col.setSelect<hkVector4ComparisonMask::MASK_ZW>(diag,perm);
		inertia.getColumn<2>().add(col);
	}
}

void HK_CALL hkInertiaTensorComputer::shiftInertiaFromCom(hkVector4Parameter shift, hkSimdRealParameter mass, hkMatrix3& inertia)
{
	shiftInertiaToCom(shift, -mass, inertia);
}


void HK_CALL hkInertiaTensorComputer::simplifyInertiaTensorToOrientedParticle(hkMatrix3 &inertia)
{
		// Take max value. This ensures that low angular impulses applied at large
		// distances from the center of mass do not result in large changes in velocity
		// (which could cause instability).
	hkVector4 temp; hkMatrix3Util::_getDiagonal(inertia, temp);
	const hkSimdReal maxDiag = temp.horizontalMax<3>();
	hkMatrix3Util::_setDiagonal(maxDiag, inertia);
}


/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////// The rest of this file contains the polytope inertia tensor code /////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

struct InternalInertiaTensorComputer
{
	void computeInertialTensorInternal(hkSimdRealParameter mass, hkSimdRealParameter density, hkVector4 &r, hkMatrix3 &J);
	void compProjectionIntegrals(const hkVector4* v);
	void compFaceIntegrals(const hkVector4* v, hkVector4Parameter n);
	void compVolumeIntegrals(const hkGeometry* geom, hkVector4Parameter shift);
	HK_FORCE_INLINE const hkVector4 forwardSubstitute(int C, hkVector4Parameter v) const;
	HK_FORCE_INLINE const hkVector4 backSubstitute(int C, hkVector4Parameter v) const;
	HK_FORCE_INLINE const hkVector4 backSubstitute(int CC, hkSimdRealParameter a, hkSimdRealParameter b, hkSimdRealParameter c) const;
	static void HK_CALL compGeometryAabb(const hkReal* vertices, int numVerts, int striding, hkVector4& centerOut, hkVector4& aabbExtentsOut );

	int m_C;   /* gamma */

	/* projection integrals */
	hkSimdReal m_P1, m_Pab, m_Paab, m_Pabb;
	hkVector4 m_P_a, m_P_b;

	/* face integrals */
	hkSimdReal m_Faab, m_Fbbc, m_Fcca;
	hkVector4 m_F_a, m_F_b, m_F_c;

	/* volume integrals */
	hkSimdReal m_T0;
	hkVector4 m_T1, m_T2, m_TP;
};

static const int InternalInertiaTensorComputer_mod3table[4]   = { 1,2,0,1 };

HK_FORCE_INLINE const hkVector4 InternalInertiaTensorComputer::backSubstitute(int CC, hkSimdRealParameter a, hkSimdRealParameter b, hkSimdRealParameter c) const
{
	hkVector4 p;

	if (CC == 0)		p.set(c,a,b,b);
	else if (CC == 1)	p.set(b,c,a,a);
	else				p.set(a,b,c,c);

	return p;
}

HK_FORCE_INLINE const hkVector4 InternalInertiaTensorComputer::backSubstitute(int C, hkVector4Parameter v) const
{
	hkVector4 p;
	
	if (C == 0)			p.setPermutation<hkVectorPermutation::ZXYW>(v);
	else if (C == 1)	p.setPermutation<hkVectorPermutation::YZXW>(v);
	else				p = v;

	return p;
}

HK_FORCE_INLINE const hkVector4 InternalInertiaTensorComputer::forwardSubstitute(int C, hkVector4Parameter v) const
{
	hkVector4 p;

	if (C == 0)			p.setPermutation<hkVectorPermutation::YZXW>(v);
	else if (C == 1)	p.setPermutation<hkVectorPermutation::ZXYW>(v);
	else				p = v;

	return p;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Mirtich inertia tensor computer code
//
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////



	/*******************************************************
    *                                                      *
	*  volInt.cpp                                          *
	*                                                      *
	*  This code computes volume integrals needed for      *
	*  determining mass properties of polyhedral bodies.   *
	*                                                      *
	*  For more information, see the paper                 *
	*                                                      *
	*  Brian Mirtich, "Fast and Accurate Computation of    *
	*  Polyhedral Mass Properties," journal of graphics    *
	*  tools, volume 1, number 1, 1996.                    *
	*                                                      *
	*  This source code is public domain, and may be used  *
	*  in any way, shape or form, free of charge.          *
	*                                                      *
	*  Copyright 1995 by Brian Mirtich                     *
	*                                                      *
	*  mirtich@cs.berkeley.edu                             *
	*  http://www.cs.berkeley.edu/~mirtich                 *
    *                                                      *
	*******************************************************/

/*
	Revision history

	26 Jan 1996	Program creation.

	 3 Aug 1996	Corrected bug arising when polyhedron density
			is not 1.0.  Changes confined to function main().
			Thanks to Zoran Popovic for catching this one.

	27 May 1997     Corrected sign error in translation of inertia
	                product terms to center of mass frame.  Changes 
			confined to function main().  Thanks to 
			Chris Hecker.
*/



	// compute various integrations over projection of face 
void InternalInertiaTensorComputer::compProjectionIntegrals(const hkVector4* v)
{
	m_P1.setZero();
	m_Pab.setZero();
	m_Paab.setZero();
	m_Pabb.setZero();

	m_P_a.setZero();
	m_P_b.setZero();

	const hkSimdReal two = hkSimdReal_2;
	const hkSimdReal three = hkSimdReal_3;
	const hkSimdReal four = hkSimdReal_4;

	// We are only dealing with triangle meshes so
	// number of vertices is always 3!
	for (int i = 0; i < 3; i++) 
	{
		const hkVector4 v_perm = forwardSubstitute(m_C, v[i]);
		const hkSimdReal a0 = v_perm.getComponent<0>();
		const hkSimdReal b0 = v_perm.getComponent<1>();

		const int next = InternalInertiaTensorComputer_mod3table[i];
		const hkVector4 v1_perm = forwardSubstitute(m_C, v[next]);
		const hkSimdReal a1 = v1_perm.getComponent<0>();
		const hkSimdReal b1 = v1_perm.getComponent<1>();

		const hkSimdReal da = a1 - a0;
		const hkSimdReal db = b1 - b0;
		const hkSimdReal a0_2 = a0 * a0; 
		const hkSimdReal a0_3 = a0_2 * a0;
		const hkSimdReal a0_4 = a0_2 * a0_2;
		const hkSimdReal b0_2 = b0 * b0; 
		const hkSimdReal b0_3 = b0_2 * b0; 
		const hkSimdReal b0_4 = b0_2 * b0_2;
		const hkSimdReal a1_2 = a1 * a1; 
		const hkSimdReal a1_3 = a1_2 * a1; 
		const hkSimdReal b1_2 = b1 * b1;
		const hkSimdReal b1_3 = b1_2 * b1;

		const hkSimdReal C1 = a1 + a0;
		const hkSimdReal Cab = three*a1_2 + two*a1*a0 + a0_2; 
		const hkSimdReal Kab = a1_2 + two*a1*a0 + three*a0_2;
		const hkSimdReal Caab = a0*Cab + four*a1_3; 
		const hkSimdReal Kaab = a1*Kab + four*a0_3;
		const hkSimdReal Cabb = four*b1_3 + three*b1_2*b0 + two*b1*b0_2 + b0_3;
		const hkSimdReal Kabb = b1_3 + two*b1_2*b0 + three*b1*b0_2 + four*b0_3;

		m_P1.add(db*C1);

		{
		   const hkSimdReal Ca = a1*C1 + a0_2; 
		   const hkSimdReal Caa = a1*Ca + a0_3; 
		   const hkSimdReal Caaa = a1*Caa + a0_4;
		   hkVector4 P_a; P_a.set(Ca,Caa,Caaa,two);
		   m_P_a.addMul(db,P_a);
		}
		{
		   const hkSimdReal Cb = b1*(b1 + b0) + b0_2; 
		   const hkSimdReal Cbb = b1*Cb + b0_3; 
		   const hkSimdReal Cbbb = b1*Cbb + b0_4;
		   hkVector4 P_b; P_b.set(Cb,Cbb,Cbbb,two);
		   m_P_b.addMul(da,P_b);
		}

		m_Pab.add(db*(b1*Cab + b0*Kab));
		m_Paab.add(db*(b1*Caab + b0*Kaab));
		m_Pabb.add(da*(a1*Cabb + a0*Kabb));
	}

	static HK_ALIGN_REAL(const hkReal scaleConst[4]) = { 1.0f/6.0f, 1.0f/12.0f, 1.0f/20.0f, 2.0f };
	const hkSimdReal invTwentyFour = hkSimdReal::fromFloat(1.0f / 24.0f);
	const hkSimdReal invSixty = hkSimdReal::fromFloat(1.0f / 60.0f);

	hkVector4 scaleA = *(const hkVector4*)&scaleConst[0];
	hkVector4 scaleB; scaleB.setNeg<4>(scaleA);

	m_P1.mul(hkSimdReal_Inv2);

	m_P_a.mul(scaleA);
	m_P_b.mul(scaleB);
	
	m_Pab.mul(invTwentyFour);
	m_Paab.mul(invSixty);
	m_Pabb.mul(-invSixty);
}

void InternalInertiaTensorComputer::compFaceIntegrals(const hkVector4* v, hkVector4Parameter n)
{
	compProjectionIntegrals(v);

	const hkSimdReal w = -n.dot<3>(v[0]);

	const hkVector4 n_perm = forwardSubstitute(m_C, n);
	hkSimdReal k1; k1.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(n_perm.getComponent<2>()); 
	const hkSimdReal k2 = k1 * k1; 
	const hkSimdReal k3 = k2 * k1; 
	const hkSimdReal k4 = k2 * k2;

	hkVector4 n_perm2; n_perm2.setMul(n_perm,n_perm);
	const hkSimdReal nA = n_perm.getComponent<0>();
	const hkSimdReal nB = n_perm.getComponent<1>();
	const hkSimdReal nA_2 = n_perm2.getComponent<0>();
	const hkSimdReal nB_2 = n_perm2.getComponent<1>();

	const hkSimdReal two = hkSimdReal_2;
	const hkSimdReal three = hkSimdReal_3;

	m_F_a.setMul(k1, m_P_a);
	m_F_b.setMul(k1, m_P_b);

	hkVector4 nAPa; nAPa.setMul(nA, m_P_a);
	hkVector4 nBPb; nBPb.setMul(nB, m_P_b);

	m_F_c.set( -k2 * (nAPa.getComponent<0>() + nBPb.getComponent<0>() + w*m_P1),
				k3 * (nA*nAPa.getComponent<1>() + two*nA*nB*m_Pab + nB*nBPb.getComponent<1>() + w*(two*(nAPa.getComponent<0>() + nBPb.getComponent<0>()) + w*m_P1)),
			   -k4 * (nA_2*nAPa.getComponent<2>() + three*nA_2*nB*m_Paab 
					+ three*nA*nB_2*m_Pabb + nB_2*nBPb.getComponent<2>()
					+ three*w*(nA*nAPa.getComponent<1>() + two*nA*nB*m_Pab + nB*nBPb.getComponent<1>())
				+ w*w*(three*(nAPa.getComponent<0>() + nBPb.getComponent<0>()) + w*m_P1)),
				two);
 
	m_Faab = k1 * m_Paab;
	m_Fbbc = -k2 * (nA*m_Pabb + nBPb.getComponent<2>() + w*m_P_b.getComponent<1>());
	m_Fcca = k3 * (nA*nAPa.getComponent<2>() + two*nA*nB*m_Paab + nB_2*m_Pabb
			+ w*(two*(nAPa.getComponent<1>() + nB*m_Pab) + w*m_P_a.getComponent<0>()));
}

void InternalInertiaTensorComputer::compVolumeIntegrals(const hkGeometry* geom, hkVector4Parameter shift)
{
	hkVector4 verts[3];
	const int numFaces = geom->m_triangles.getSize(); 

	m_T0.setZero();
	m_T1.setZero();
    m_T2.setZero();
	m_TP.setZero();

	for (int i = 0; i < numFaces; i++) 
	{
		const hkGeometry::Triangle& tindices = geom->m_triangles[i];
		verts[0].setAdd(geom->m_vertices[tindices.m_a],shift);
		verts[1].setAdd(geom->m_vertices[tindices.m_b],shift);
		verts[2].setAdd(geom->m_vertices[tindices.m_c],shift);
		hkVector4 normal;
		{
			hkVector4 sub1;
			hkVector4 sub2;	
			sub1.setSub(verts[1],verts[0]);
			sub2.setSub(verts[2],verts[0]);
			normal.setCross(sub1,sub2);
		}
		
		const hkSimdReal area2 = normal.lengthSquared<3>();

		if (area2.isGreaterZero())	// ignore zero area faces
		{

			// okay this is twice the area...but we don't care
			const hkSimdReal invlength = area2.sqrtInverse<HK_ACC_FULL,HK_SQRT_IGNORE>();
			normal.mul(invlength);
			
			m_C = normal.getIndexOfMaxAbsComponent<3>();
		
			compFaceIntegrals(verts,normal);
			
			HK_TRANSPOSE3(m_F_a,m_F_b,m_F_c);

			m_T0.add( normal.getComponent<0>() * m_F_a.getComponent(2-m_C) );
			{
				const hkVector4 p = backSubstitute(m_C, m_F_b);
				m_T1.addMul(normal, p);
			}
			{
				const hkVector4 p = backSubstitute(m_C, m_F_c);
				m_T2.addMul(normal, p);
			}
			{
				const hkVector4 p = backSubstitute(m_C, m_Faab, m_Fbbc, m_Fcca);
				m_TP.addMul(normal, p);
			}
		}
	}

	m_T1.mul(hkSimdReal_Inv2);
	m_T2.mul(hkSimdReal_Inv3);
	m_TP.mul(hkSimdReal_Inv2);

#if defined(__MWERKS__)
#pragma optimization_level 0
#endif

}

#if defined(__MWERKS__)
#pragma optimization_level 4
#endif


void InternalInertiaTensorComputer::computeInertialTensorInternal(hkSimdRealParameter mass, hkSimdRealParameter density,
												   hkVector4 &r, hkMatrix3 &J)
{
	/* compute center of mass */
	hkSimdReal invT0; invT0.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(m_T0);
	r.setMul(m_T1, invT0);

	/* translate inertia tensor to center of mass */
	hkVector4 diag;
	{
		hkVector4 t2YZX; t2YZX.setPermutation<hkVectorPermutation::YZXW>(m_T2);
		hkVector4 t2ZXY; t2ZXY.setPermutation<hkVectorPermutation::ZXYW>(m_T2);
		diag.setAdd(t2YZX, t2ZXY);
		diag.mul(density);

		hkVector4 r2; r2.setMul(r,r); 
		hkVector4 r2YXX; r2YXX.setPermutation<hkVectorPermutation::YXXY>(r2);
		hkVector4 r2ZZY; r2ZZY.setPermutation<hkVectorPermutation::ZZYY>(r2);
		r2.setAdd(r2YXX,r2ZZY);
		r2.mul(mass);
		diag.sub(r2);

		diag.zeroComponent<3>(); // for clearing the columns 4th components below
	}
	hkVector4 offDiag;
	{
		hkVector4 tp; tp.setMul(-density, m_TP);
		hkVector4 rYZX; rYZX.setPermutation<hkVectorPermutation::YZXW>(r);
		hkVector4 mr; mr.setMul(r, rYZX); 
		mr.mul(mass);
		offDiag.setAdd(tp,mr);
	}

	{
		hkVector4 perm; perm.setPermutation<hkVectorPermutation::XXZZ>(offDiag);
		J.getColumn<0>().setSelect<hkVector4ComparisonMask::MASK_XW>(diag, perm);
	}
	{
		hkVector4 perm; perm.setPermutation<hkVectorPermutation::XXYY>(offDiag);
		J.getColumn<1>().setSelect<hkVector4ComparisonMask::MASK_YW>(diag, perm);
	}
	{
		hkVector4 perm; perm.setPermutation<hkVectorPermutation::ZYZW>(offDiag);
		J.getColumn<2>().setSelect<hkVector4ComparisonMask::MASK_ZW>(diag, perm);
	}
}


// This computes an AABB which can be used to ensure that the geometry is "roughly" centered
// around the origin. This enables us to do the computations on a the "shifted" 
// geometry, and avoid bad numerical roundoff errors.
// This routine expects SIMD aligned 4-component vectors.
void HK_CALL InternalInertiaTensorComputer::compGeometryAabb(const hkReal* vertices, int numVerts, int striding, hkVector4& centerOut, hkVector4& aabbExtentsOut )
{
	centerOut.setZero();
	aabbExtentsOut.setZero();

	hkVector4 minP = hkVector4::getConstant<HK_QUADREAL_MAX>();
	hkVector4 maxP;	maxP.setNeg<4>(minP);
	hkVector4 v;

	for(int i = 0; i< numVerts; i++)
	{
		v.load<4,HK_IO_NOT_CACHED>(hkAddByteOffsetConst(vertices,i*striding));
		minP.setMin(v, minP);
		maxP.setMax(v, maxP);
	}

	if(numVerts != 0)
	{
		centerOut.setInterpolate( minP, maxP, hkSimdReal_Inv2 );
		aabbExtentsOut.setSub( maxP, minP );
	}
}


// WARNING: This function assumes the geometry is properly closed.
void HK_CALL hkInertiaTensorComputer::computeGeometryVolumeMassProperties(const hkGeometry* geom, hkReal mass, hkMassProperties &result)
{
	HK_ASSERT2(0x7df8b01c,  mass > hkReal(0), "Cannot calculate mass properties with zero mass or less." );

	// safety check for mass
	hkSimdReal m; m.load<1>(&mass);
	m.setSelect(m.lessEqualZero(), hkSimdReal_1, m);

	hkVector4 aabbCenter;
	hkVector4 aabbExtents;
	InternalInertiaTensorComputer::compGeometryAabb((const hkReal*)geom->m_vertices.begin(), geom->m_vertices.getSize(), hkSizeOf(hkVector4), aabbCenter, aabbExtents );

		//
		// Calculate an AABB inertia as a safety check
		//
	hkMassProperties aabbInertia;
	{
		HK_ON_DEBUG(hkResult res_box = ) computeBoxVolumeMassProperties( aabbExtents, m.getReal(), aabbInertia );
		HK_ASSERT2(0x28377b, res_box == HK_SUCCESS, "failed to compute aabb inertia");
		aabbInertia.m_centerOfMass = aabbCenter;
	}


		//
		//	Calculate the volume integrals
		//
	InternalInertiaTensorComputer computer;
	{
		hkVector4 shift; shift.setNeg<4>( aabbCenter );
		computer.compVolumeIntegrals(geom, shift);
	}

	hkSimdReal vol = computer.m_T0;

	if (vol.isLessEqualZero())
	{
		HK_WARN(0x47b91356, "Cannot calculate mass properties of a hkGeometry with zero volume. Using AABB inertia instead.");
		result = aabbInertia;
		return;
	}

	hkSimdReal density; density.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(m,vol);

	result.m_volume = vol.getReal();
	result.m_mass   = m.getReal();

	hkMatrix3& J = result.m_inertiaTensor;
	{
		computer.computeInertialTensorInternal(m, density, result.m_centerOfMass, J);
		// Must subtract shift (since we forced readPolyhedron to construct geometry data with
		// shift added to each vertex, hence VOL, IT are the same, but C.O.M for the "unshifted"
		// geometry is equal to C.O.M(shifted) - "shift". This has NOTHING to do with any body/local/world transforms.
		result.m_centerOfMass.add(aabbCenter);
	}

	hkVector4Comparison jLessA;
	{
		// check for degenerated inertias
		hkVector4 diagJ; hkMatrix3Util::_getDiagonal(J, diagJ);
		hkVector4 diagA; hkMatrix3Util::_getDiagonal(aabbInertia.m_inertiaTensor, diagA);

		diagA.mul(hkSimdReal::fromFloat(0.1f));
		jLessA = diagJ.less(diagA);

		diagJ.setSelect(jLessA, diagA, diagJ);
		result.m_centerOfMass.setSelect(jLessA, aabbCenter, result.m_centerOfMass);

		hkMatrix3Util::_setDiagonalOnly(diagJ, J);
	}
	{
		// clear r/c of components set to aabb inertia
		hkVector4 zero; zero.setZero();
		hkVector4 c0; c0.setMax(zero, J.getColumn<0>());
		hkVector4 c1; c1.setMax(zero, J.getColumn<1>());
		hkVector4 c2; c2.setMax(zero, J.getColumn<2>());

		
		const hkVector4Comparison::Mask mask = jLessA.getMask();
		if (mask & hkVector4ComparisonMask::MASK_X)
		{
			J(1,0) = J(0,1) = c0(1);
			J(2,0) = J(0,2) = c0(2);
		}
		if (mask & hkVector4ComparisonMask::MASK_Y)
		{
			J(0,1) = J(1,0) = c1(0);
			J(2,1) = J(1,2) = c1(2);
		}
		if (mask & hkVector4ComparisonMask::MASK_Z)
		{
			J(0,2) = J(2,0) = c2(0);
			J(1,2) = J(2,1) = c2(1);
		}
	}
}

// WARNING: This function assumes the geometry is properly closed.
hkResult HK_CALL hkInertiaTensorComputer::computeGeometryVolumeMassPropertiesChecked(const hkGeometry* geom, hkReal mass, hkMassProperties &result)
{
	HK_ASSERT2(0x7df8b01c,  mass > hkReal(0), "Cannot calculate mass properties with zero mass or less." );

	// safety check for mass
	hkSimdReal m; m.load<1>(&mass);
	if (m.isLessEqualZero())
	{
		return HK_FAILURE;
	}

	hkVector4 aabbCenter;
	hkVector4 aabbExtents;
	InternalInertiaTensorComputer::compGeometryAabb((const hkReal*)geom->m_vertices.begin(), geom->m_vertices.getSize(), hkSizeOf(hkVector4), aabbCenter, aabbExtents );

	//
	// Calculate an AABB inertia as a safety check
	//
	hkMassProperties aabbInertia;
	{
		HK_ON_DEBUG(hkResult res_box = ) computeBoxVolumeMassProperties( aabbExtents, mass, aabbInertia );
		HK_ASSERT2(0x28377b, res_box == HK_SUCCESS, "failed to compute aabb inertia");
		aabbInertia.m_centerOfMass = aabbCenter;
	}


	//
	//	Calculate the volume integrals
	//
	InternalInertiaTensorComputer computer;
	{
		hkVector4 shift; shift.setNeg<4>( aabbCenter );
		computer.compVolumeIntegrals(geom, shift);
	}

	const hkSimdReal vol = computer.m_T0;

	if (vol.isLessEqualZero())
	{
		return HK_FAILURE;
	}

	hkSimdReal density; density.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(m,vol);

	result.m_volume = vol.getReal();
	result.m_mass   = mass;

	hkMatrix3& J = result.m_inertiaTensor;
	{
		computer.computeInertialTensorInternal(m, density, result.m_centerOfMass, J);
		result.m_centerOfMass.add(aabbCenter);
	}

	hkVector4Comparison jLessA;
	{
		// check for degenerated inertias
		hkVector4 diagJ; hkMatrix3Util::_getDiagonal(J, diagJ);
		hkVector4 diagA; hkMatrix3Util::_getDiagonal(aabbInertia.m_inertiaTensor, diagA);

		diagA.mul(hkSimdReal::fromFloat(0.1f));
		jLessA = diagJ.less(diagA);

		diagJ.setSelect(jLessA, diagA, diagJ);
		result.m_centerOfMass.setSelect(jLessA, aabbCenter, result.m_centerOfMass);

		hkMatrix3Util::_setDiagonalOnly(diagJ, J);
	}
	{
		// clear r/c of components set to aabb inertia
		hkVector4 zero; zero.setZero();
		hkVector4 c0; c0.setMax(zero, J.getColumn<0>());
		hkVector4 c1; c1.setMax(zero, J.getColumn<1>());
		hkVector4 c2; c2.setMax(zero, J.getColumn<2>());

		
		const hkVector4Comparison::Mask mask = jLessA.getMask();
		if (mask & hkVector4ComparisonMask::MASK_X)
		{
			J(1,0) = J(0,1) = c0(1);
			J(2,0) = J(0,2) = c0(2);
		}
		if (mask & hkVector4ComparisonMask::MASK_Y)
		{
			J(0,1) = J(1,0) = c1(0);
			J(2,1) = J(1,2) = c1(2);
		}
		if (mask & hkVector4ComparisonMask::MASK_Z)
		{
			J(0,2) = J(2,0) = c2(0);
			J(1,2) = J(2,1) = c2(1);
		}
	}

	return HK_SUCCESS;
}



void hkMassProperties::scaleToMass( hkSimdRealParameter newMass )
{
	HK_ASSERT2( 0xf032de12, m_mass > hkReal(0), "Cannot scale the mass of a zero mass body" );
	hkSimdReal m; m.load<1>(&m_mass);
	hkSimdReal f; f.setDiv<HK_ACC_FULL,HK_DIV_IGNORE>(newMass, m);
	newMass.store<1>(&m_mass);
	m_inertiaTensor.mul(f);
}

void hkMassProperties::scaleToDensity( hkSimdRealParameter density )
{
	hkSimdReal newMass;
	newMass.load<1>(&m_volume); newMass.mul(density);
	scaleToMass( newMass );
}



// Compute mass properties of the convex hull's Aabb. This approach avoids linking hkgpConvexHull
hkResult HK_CALL hkInertiaTensorComputer::computeApproximateConvexHullMassProperties(const hkStridedVertices& vertices, hkReal radius, hkMassProperties& result)
{
	hkVector4 aabbCenter;
	hkVector4 aabbExtents;
	InternalInertiaTensorComputer::compGeometryAabb(vertices.m_vertices, vertices.getSize(), vertices.m_striding, aabbCenter, aabbExtents );

	hkVector4 paddedHalfExtents;
	paddedHalfExtents.setAll( radius );
	paddedHalfExtents.addMul( hkSimdReal_Inv2, aabbExtents );

	hkResult res = computeBoxVolumeMassProperties( paddedHalfExtents, hkReal(1), result );
	result.m_centerOfMass = aabbCenter;
	result.m_mass = hkReal(1);

	return res;
}


void HK_CALL hkInertiaTensorComputer::convertInertiaTensorToPrincipleAxis( hkMatrix3& inertia, hkRotation& principleAxisOut )
{
	hkVector4 eigenVal;
	inertia.diagonalizeSymmetricApproximation( principleAxisOut, eigenVal );
	eigenVal.setMax( eigenVal, hkVector4::getConstant<HK_QUADREAL_EPS>());	
	hkMatrix3Util::_setDiagonal( eigenVal, inertia );

	//
	// renormalize output
	//
	principleAxisOut.getColumn<0>().normalize<3>();

	principleAxisOut.getColumn<1>().setCross( principleAxisOut.getColumn<2>(), principleAxisOut.getColumn<0>() );
	principleAxisOut.getColumn<1>().normalize<3>();

	principleAxisOut.getColumn<2>().setCross( principleAxisOut.getColumn<0>(), principleAxisOut.getColumn<1>() );
	principleAxisOut.getColumn<2>().normalize<3>();

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

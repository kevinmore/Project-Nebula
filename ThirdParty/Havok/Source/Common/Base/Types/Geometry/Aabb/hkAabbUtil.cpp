/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

namespace hkAabbUtil
{

//
//	The maximum extents of a hkAabbUin32 (in int space): we cannot use the full
//  32 bit as we have to use the sign bit (without the overflow bit) to compare
//  two values;
//

hkQuadReal hkAabbUint32MaxVal = HK_QUADREAL_CONSTANT(
	hkReal(AABB_UINT32_MAX_FVALUE), hkReal(AABB_UINT32_MAX_FVALUE),
	hkReal(AABB_UINT32_MAX_FVALUE), hkReal(AABB_UINT32_MAX_FVALUE)
);

#if !defined(HK_AABBUTIL_convertAabbToUint32)
		HK_COMPILE_TIME_ASSERT( HK_OFFSET_EQUALS( hkAabbUint32, m_min, 0 ) );
		HK_COMPILE_TIME_ASSERT( HK_OFFSET_EQUALS( hkAabbUint32, m_max, 0x10 ) );
#endif


void HK_CALL calcAabb( const hkReal* vertexArray, int numVertices, int striding, hkAabb& aabbOut )
{
	aabbOut.setEmpty();

	if ( numVertices <= 0 )
	{
		return;
	}

	hkVector4 v; v.setZero();
	for (int i = 0; i < numVertices; i++)
	{
		v.load<3,HK_IO_NATIVE_ALIGNED>( hkAddByteOffsetConst(vertexArray, i*striding) );
		aabbOut.includePoint(v);
	}

	aabbOut.m_min.zeroComponent<3>();
	aabbOut.m_max.zeroComponent<3>();
}

void HK_CALL calcAabb( const hkVector4* vertexArray, int numVertices, hkAabb& aabbOut )
{
	aabbOut.setEmpty();
	
	if ( numVertices <= 0 )
	{
		return;
	}

	for (int i = 0; i < numVertices; i++)
	{
		const hkVector4& v = vertexArray[i];
		aabbOut.includePoint(v);
	}

	aabbOut.m_min.zeroComponent<3>();
	aabbOut.m_max.zeroComponent<3>();
}

void HK_CALL calcAabb(const hkTransform& BvToWorld, const hkAabb& aabb, hkSimdRealParameter extraRadius, hkAabb& aabbOut)
{
	hkVector4 center;		aabb.getCenter(center);
	hkVector4 halfExtents;	aabb.getHalfExtents(halfExtents);
	
	calcAabb(BvToWorld, halfExtents, center, extraRadius, aabbOut);
}


void HK_CALL calcAabb(const hkTransform& BvToWorld, const hkAabb& aabb, hkAabb& aabbOut)
{
	hkVector4 center;		aabb.getCenter(center);
	hkVector4 halfExtents;	aabb.getHalfExtents(halfExtents);

	calcAabb(BvToWorld, halfExtents, center, aabbOut);
}


void HK_CALL calcAabb(const hkQsTransform& bvToWorld, const hkAabb& aabb, hkAabb& aabbOut)
{
	// Scale center
	hkTransform bvToWorldNoScale; bvToWorld.copyToTransformNoScale(bvToWorldNoScale);
	const hkVector4& scale = bvToWorld.getScale();
	hkVector4 center; aabb.getCenter(center); 
	center.mul(scale);

	// Scale half extents
	hkVector4 halfExtents;	aabb.getHalfExtents(halfExtents);
	halfExtents.mul(scale);
	halfExtents.setAbs(halfExtents);

	// Apply transform without scale
	calcAabb(bvToWorldNoScale, halfExtents, center, aabbOut);
}


void HK_CALL calcAabb( const hkQTransform& bvToWorld, const hkAabb& aabb, hkAabb& aabbOut )
{
	// Scale center
	hkTransform bBvToWorldNoScale; bBvToWorldNoScale.set( bvToWorld.getRotation(), bvToWorld.getTranslation() );
	hkVector4 center; aabb.getCenter(center); 

	// Scale half extents
	hkVector4 halfExtents;	aabb.getHalfExtents(halfExtents);
	halfExtents.setAbs(halfExtents);

	// Apply transform without scale
	calcAabb(bBvToWorldNoScale, halfExtents, center, aabbOut);
}


//
void HK_CALL calcAabb( const hkVector4*const* vectorArray, int numVertices, hkAabb& aabbOut )
{
	aabbOut.setEmpty();

	if(numVertices <= 0)
	{
		return;
	}

	for(int i=0;i<numVertices;++i)
	{
		aabbOut.includePoint(*vectorArray[i]);
	}

	aabbOut.m_min.zeroComponent<3>();
	aabbOut.m_max.zeroComponent<3>();
}

#if defined(HK_PLATFORM_SPU)
void HK_CALL sweepOffsetAabb(const OffsetAabbInput& input, const hkAabb& aabbIn, hkAabb& aabbOut) 
{
	hkAabbUtil_sweepOffsetAabb(input, aabbIn, aabbOut);
}
#endif

void HK_CALL initOffsetAabbInput(const hkMotionState* motionState, OffsetAabbInput& input)
{
	input.m_motionState = motionState;
	input.m_endTransformInv.setInverse(motionState->getTransform());

	const hkSweptTransform& swept = motionState->getSweptTransform();
	swept._approxTransformAt(swept.getBaseTimeSr(), input.m_startTransform);

	if (swept.getInvDeltaTimeSr().isNotEqualZero())
	{
		hkSimdReal deltaTime; deltaTime.setReciprocal(swept.getInvDeltaTimeSr());
		const hkSimdReal deltaAngle = motionState->m_deltaAngle.getComponent<3>();
		const hkSimdReal pi_8 = hkSimdReal::fromFloat(HK_REAL_PI * 0.125f);
		const hkSimdReal pi_4 = hkSimdReal_PiOver4;
		if (deltaAngle <= pi_8)
		{
			// Just one inter transform
			hkSimdReal invCosApprox; invCosApprox.setReciprocal( hkSimdReal_1 - deltaAngle*deltaAngle*hkSimdReal_Inv2 );
			input.m_transforms[0] = input.m_startTransform;
			input.m_transforms[0].getTranslation().setComponent<3>(invCosApprox);
			input.m_numTransforms = 1;
			return;
		}

		if (deltaAngle <= pi_4)
		{
			// Just one inter transform
			const hkSimdReal time = swept.getBaseTimeSr() + hkSimdReal_Inv2 * deltaTime;
			motionState->getSweptTransform()._approxTransformAt(time, input.m_transforms[0]);
			// extended arm for the in-between transforms (cos(22.5deg)
			hkSimdReal invCosApprox; invCosApprox.setReciprocal( hkSimdReal_1 - deltaAngle*deltaAngle*hkSimdReal_Inv4*hkSimdReal_Inv2 );
			input.m_transforms[0].getTranslation().setComponent<3>(invCosApprox);
			input.m_numTransforms = 1;
			return;
		}

		{
			const hkSimdReal parts = (deltaAngle + pi_8) / pi_8;
			hkSimdReal partsInv; partsInv.setReciprocal(parts);
			input.m_numTransforms = 0;
			const hkSimdReal cst = hkSimdReal::fromFloat(1.0824f);
			for (hkSimdReal p = hkSimdReal_1; p < parts; p.add(hkSimdReal_2))
			{
				const hkSimdReal time = swept.getBaseTimeSr() + (p*partsInv) * deltaTime;
				HK_ASSERT2(0xad7644aa, input.m_numTransforms < 4, "The fixed-capacity Transforms array to small. Make it larger.");
				hkTransform& t = input.m_transforms[input.m_numTransforms];
				input.m_numTransforms = input.m_numTransforms + 1;
				motionState->getSweptTransform()._approxTransformAt(time, t);
				t.getTranslation().setComponent<3>(cst);
			}
		}
	}
	else
	{
		input.m_numTransforms = 0;
	}
}

} // namespace hkAabbUtil

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

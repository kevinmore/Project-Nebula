/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/Math/Vector/hkIntVector.h>

HK_FORCE_INLINE void hknpDeactivationStateUtil::storeReferencePosition(hkVector4Parameter refPosition, hkSimdRealParameter invBlockSize, hkUint32& compressedPosition)
{
	//10-10-10 compression
	hkSimdReal packResolution; packResolution.setFromFloat(1024.0f);
	hkSimdReal posToIntPosFactor; posToIntPosFactor.setMul(packResolution, invBlockSize);

	hkVector4 fPos; fPos.setMul(posToIntPosFactor, refPosition);

	hkIntVector iPos; iPos.setConvertF32toS32(fPos);

	//set the first 32bit int of iPos to (z << 20) | (y << 10) | x (every component has 10 bits)
	static const HK_ALIGN16( hkUint32 mask0x3ff[4] ) = { 0x3ff, 0x3ff, 0x3ff, 0x3ff };
	iPos.setAnd(iPos, *(hkIntVector*)mask0x3ff);

	hkIntVector iPosShift10; iPosShift10.setShiftLeft32<10>(iPos);
	hkIntVector iPosShift20; iPosShift20.setShiftLeft32<20>(iPos);
	iPosShift10.setPermutation<hkVectorPermutation::YZWX>(iPosShift10);
	iPosShift20.setPermutation<hkVectorPermutation::ZWXY>(iPosShift20);
	iPos.setOr(iPos,iPosShift10);
	iPos.setOr(iPos,iPosShift20);

	iPos.storeNotAligned<1>(&compressedPosition);

	//hkUint32 x = iPos.getU32<0>() & 0x3ff;
	//hkUint32 y = iPos.getU32<1>() & 0x3ff;
	//hkUint32 z = iPos.getU32<2>() & 0x3ff;

	//hkUint32 packedPosition = (z << 20) | (y << 10) | x;

	//m_referencePosition = packedPosition;
}

HK_FORCE_INLINE void hknpDeactivationStateUtil::extractReferencePosition(hkVector4& refPositionOut, hkSimdRealParameter invBlockSize, hkUint32& compressedPosition)
{
	hkSimdReal invPackResolution; invPackResolution.setFromFloat(0.0009765625f);
	hkSimdReal intToFloat; intToFloat.setDiv<HK_ACC_23_BIT, HK_DIV_IGNORE>(invPackResolution, invBlockSize);

	//hkUint32 components[4];
	//components[0] = m_referencePosition & 0x3ff;
	//components[1] = (m_referencePosition >> 10) & 0x3ff;
	//components[2] = (m_referencePosition >> 20);

	//hkIntVector iPos;
	//iPos.loadNotAligned<4>( components );

	hkIntVector iPos;
	iPos.loadNotAligned<1>( &compressedPosition );

	hkIntVector iPosY; iPosY = iPos;
	hkIntVector iPosZ; iPosZ = iPos;

	iPosY.setShiftRight32<10>(iPosY);
	iPosZ.setShiftRight32<20>(iPosZ);

	static const HK_ALIGN16( hkUint32 mask0x3ff[4] ) = { 0x3ff, 0x3ff, 0x3ff, 0x3ff };
	iPos.setAnd(iPos, *(hkIntVector*)mask0x3ff);
	iPosY.setAnd(iPosY, *(hkIntVector*)mask0x3ff);
	iPosZ.setAnd(iPosZ, *(hkIntVector*)mask0x3ff);

	iPos.setMergeHead32(iPos,iPosZ);
	iPos.setMergeHead32(iPos,iPosY);

	hkVector4 fPos;	iPos.convertS32ToF32(fPos);

	refPositionOut.setMul(intToFloat, fPos);
}


HK_FORCE_INLINE void hknpDeactivationStateUtil::positionAfterCompression(hkVector4Parameter positionIn, hkVector4& positionOut, hkSimdRealParameter invBlockSize)
{
	//do the packing and unpacking in sequence, without leaving simd
	hkSimdReal packResolution; packResolution.setFromFloat(1024.0f);
	hkSimdReal invPackResolution; invPackResolution.setFromFloat(0.0009765625f);
	hkSimdReal posToIntFactor; posToIntFactor.setMul(packResolution, invBlockSize);
	hkSimdReal intToPosFactor; intToPosFactor.setDiv<HK_ACC_23_BIT, HK_DIV_IGNORE>(invPackResolution, invBlockSize);

	hkVector4 fPos; 	fPos.setMul(positionIn, posToIntFactor);

	hkIntVector iPos; 	iPos.setConvertF32toS32(fPos);

	//hkIntVector mask0x3ff; mask0x3ff.setAll(0x3ff);
	static const HK_ALIGN16( hkUint32 mask0x3ff[4] ) = { 0x3ff, 0x3ff, 0x3ff, 0x3ff };
	iPos.setAnd(iPos, *((hkIntVector*)mask0x3ff) );

	iPos.convertS32ToF32(fPos);

	positionOut.setMul(fPos, intToPosFactor);
}

HK_FORCE_INLINE void hknpDeactivationStateUtil::storeReferenceOrientation(const hkQuaternion& refOrientation, hkUint32& compressedOrientation)
{
	//m_referenceOrientation = hkVector4Util::packQuaternionIntoInt32(refOrientation.m_vec);

	hkVector4 packRange; packRange.setAll(1.1f);
	hkVector4 resolution; resolution.setAll(128.0f);
	hkVector4 scale; scale.setDiv(resolution, packRange);

	hkVector4 x = scale;
	x.mul(refOrientation.m_vec);
	//x.add( hkVector4::getConstant<HK_QUADREAL_INV_2>() );
	hkIntVector iRot; iRot.setConvertF32toS32(x);

	static const HK_ALIGN16( hkUint32 offset0x80[4] ) = { 0x7F, 0x7F, 0x7F, 0x7F };
	iRot.setAddU32(iRot, *(hkIntVector*)offset0x80);
	iRot.setConvertSaturateS32ToS16(iRot, iRot);
	iRot.setConvertSaturateS16ToU8(iRot, iRot);
	iRot.storeNotAligned<1>(&compressedOrientation);

	//hkUint32 a = iRot.getU32<0>();
	//hkUint32 b = iRot.getU32<1>();
	//hkUint32 c = iRot.getU32<2>();
	//hkUint32 d = iRot.getU32<3>();
	//a = (a + 0x80) & 0xff;
	//b = (b + 0x80) & 0xff;
	//c = (c + 0x80) & 0xff;
	//d = (d + 0x80) & 0xff;
	//compressedOrientation = a | (b << 8) | (c << 16) | (d << 24);
}

HK_FORCE_INLINE void hknpDeactivationStateUtil::extractReferenceOrientation(hkQuaternion& refOrientationOut, hkUint32& compressedOrientation)
{
	//hkVector4Util::unPackInt32IntoQuaternion( m_referenceOrientation, refOrientationOut.m_vec );

	hkVector4 packRange; packRange.setAll(1.1f);
	hkVector4 resolution; resolution.setAll(128.0f);
	hkVector4 scale; scale.setDiv(packRange, resolution);

	//hkInt32 ivalue = m_referenceOrientation;

	//hkInt32 a = ivalue;
	//hkInt32 b = ivalue >> 8;
	//hkInt32 c = ivalue >> 16;
	//hkInt32 d = ivalue >> 24;
	//a = (a & 0xff) - 0x80;
	//b = (b & 0xff) - 0x80;
	//c = (c & 0xff) - 0x80;
	//d = (d & 0xff) - 0x80;
	//hkIntVector iRot; iRot.set(a,b,c,d);

	hkIntVector iRot;
	iRot.loadNotAligned<1>( &compressedOrientation );

	iRot.setSplit8To32(iRot);

	static const HK_ALIGN16( hkUint32 offset0x80[4] ) = { 0x7F, 0x7F, 0x7F, 0x7F };
	iRot.setSubU32(iRot, *(hkIntVector*)offset0x80);

	hkVector4 fOrient;
	iRot.convertS32ToF32(fOrient);
	fOrient.mul(scale);

	refOrientationOut.m_vec = fOrient;
}

HK_FORCE_INLINE void hknpDeactivationStateUtil::orientationAfterCompression(const hkQuaternion& orientationIn, hkQuaternion& orientationOut)
{
	//do the packing and unpacking in sequence, without leaving simd
	hkSimdReal packRange; packRange.setFromFloat(1.1f);
	hkSimdReal resolution; resolution.setFromFloat(128.0f);
	hkSimdReal scale;	 scale.setDiv<HK_ACC_23_BIT, HK_DIV_IGNORE>(resolution, packRange);
	hkSimdReal scaleInv; scaleInv.setDiv<HK_ACC_23_BIT, HK_DIV_IGNORE>(packRange, resolution);

	hkVector4 fOrient;
	fOrient.setMul(orientationIn.m_vec, scale);
	//fOrient.add( hkVector4::getConstant<HK_QUADREAL_INV_2>() );
	hkIntVector iOrient; iOrient.setConvertF32toS32(fOrient);

	//static const HK_ALIGN16( hkUint32 mask0xff[4] ) = { 0xff, 0xff, 0xff, 0xff };
	//static const HK_ALIGN16( hkUint32 offset0x80[4] ) = { 0x80, 0x80, 0x80, 0x80 };
	//iOrient.setAddU32(iOrient, *((hkIntVector*)offset0x80));
	//iOrient.setAnd(iOrient, *((hkIntVector*)mask0xff));
	//iOrient.setSubU32(iOrient, *((hkIntVector*)offset0x80));

	iOrient.convertS32ToF32(fOrient);
	orientationOut.m_vec.setMul(fOrient, scaleInv);
}


HK_FORCE_INLINE void hknpDeactivationStateUtil::accumVelocityChange(hkSimdRealParameter velocityDiff, hkInt16& accumVelocityDiff)
{
	hkIntVector loadedAccumVelocityDiff; loadedAccumVelocityDiff.loadNotAligned<1>( (hkUint32*)&accumVelocityDiff ); //also load the surrounding bytes

	static const HK_ALIGN16( hkUint32 offset0xFFFFMask[4] ) = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF };
	hkIntVector accumVelDiffVecI; accumVelDiffVecI.setAnd(loadedAccumVelocityDiff, *(hkIntVector*)offset0xFFFFMask);
	accumVelDiffVecI.setSignExtendS16ToS32(accumVelDiffVecI);

	hkSimdReal scale; scale.setFromFloat(32767.0f); //the maximum int16 value
	hkSimdReal toCompressedVelDiff; toCompressedVelDiff.setAddMul(hkSimdReal::getConstant<HK_QUADREAL_INV_2>(), velocityDiff, scale);//add 0.5 to account for truncation applied to toCompressedVelDiff; desired behaviour is round to nearest integer

	//add in 32 bits and clamp to 16
	hkVector4 compressedVelDiffVec4; compressedVelDiffVec4.setAll(toCompressedVelDiff);
	hkIntVector compressedVelDiffVecI; compressedVelDiffVecI.setConvertF32toS32(compressedVelDiffVec4);
	hkIntVector newAccumVelDiffVecI; newAccumVelDiffVecI.setAddU32(accumVelDiffVecI, compressedVelDiffVecI);
	newAccumVelDiffVecI.setConvertSaturateS32ToS16(newAccumVelDiffVecI, newAccumVelDiffVecI);

	//mask again
	newAccumVelDiffVecI.setAnd(newAccumVelDiffVecI, *(hkIntVector*)offset0xFFFFMask);

	loadedAccumVelocityDiff.setAndNot(loadedAccumVelocityDiff,*(hkIntVector*)offset0xFFFFMask);
	newAccumVelDiffVecI.setOr(newAccumVelDiffVecI, loadedAccumVelocityDiff);
	newAccumVelDiffVecI.storeNotAligned<1>((hkUint32*)&accumVelocityDiff);//goes over the boundaries, but the original contents are still in the overflow bytes
}

HK_FORCE_INLINE void hknpDeactivationStateUtil::storeAccelerationDirection(hkVector4Parameter accelerationDir, hkUint8& accelerationDirection)
{
	//HK_ASSERT(0x76678765,accelerationDir.isNormalized<3>());
	const hkVector4& normalizedAcceleration = accelerationDir;

#if 0
	hkReal almostOne;
	almostOne = 1.0f - HK_REAL_EPSILON;
	hkReal toCompressedRangeX;
	toCompressedRangeX = 8.0f; //16 discrete steps total
	hkReal toCompressedRangeY;
	toCompressedRangeY = 4.0f; //8 discrete steps total

	hkReal xDirForCompression = normalizedAcceleration.getComponent<0>().getReal();
	hkReal yDirForCompression = normalizedAcceleration.getComponent<1>().getReal();
	hkUint32 compressedXDir = hkUint32(xDirForCompression * almostOne * toCompressedRangeX + toCompressedRangeX);
	hkUint32 compressedYDir = hkUint32(yDirForCompression * almostOne * toCompressedRangeY + toCompressedRangeY);
	hkUint32 compressedZDir = normalizedAcceleration.getComponent<2>().isLessZero() ? 0 : 1; //last remaining bit is for the direction
	HK_ASSERT(0x76678766, compressedXDir <= 0xF);
	HK_ASSERT(0x76678766, compressedYDir <= 0x7);
	accelerationDirection = hkUint8(compressedXDir | (compressedYDir << 4) | (compressedZDir << 7));
#else
	hkReal thetaForCompression = hkMath::acos(normalizedAcceleration.getComponent<2>().getReal());
	hkReal phiForCompression = hkMath::atan2(normalizedAcceleration.getComponent<1>().getReal(), normalizedAcceleration.getComponent<0>().getReal());

	hkReal almostOne;
	almostOne = 1.0f - HK_REAL_EPSILON;
	hkReal toCompressedRange;
	toCompressedRange = 16.0f; //16 discrete steps total
	hkReal twoPi = 2.0f * HK_REAL_PI;

	thetaForCompression = thetaForCompression / HK_REAL_PI;
	phiForCompression = (phiForCompression < 0.0f ? phiForCompression + twoPi : phiForCompression) / twoPi;

	hkUint32 compressedTheta = hkUint32(thetaForCompression * almostOne * toCompressedRange);
	hkUint32 compressedPhi = hkUint32(phiForCompression * almostOne * toCompressedRange);
	HK_ASSERT(0x76678766, compressedTheta <= 0xF);
	HK_ASSERT(0x76678766, compressedPhi <= 0xF);
	accelerationDirection = hkUint8(compressedTheta | (compressedPhi << 4));
#endif

}

HK_FORCE_INLINE void hknpDeactivationStateUtil::extractAccelerationDirection(hkVector4& accelerationDirOut, hkUint8& accelerationDirection)
{
#if 0
	hkReal almostOne;
	almostOne = 1.0f - HK_REAL_EPSILON;
	hkReal toCompressedRangeX;
	toCompressedRangeX = 8.0f; //16 discrete steps total
	hkReal toCompressedRangeY;
	toCompressedRangeY = 4.0f; //8 discrete steps total

	hkReal compressedXDir = hkReal(accelerationDirection & 0xF);
	hkReal compressedYDir = hkReal((accelerationDirection & 0x70) >> 4);
	hkReal compressedZDir = hkReal(((accelerationDirection & 0x80) >> 7) == 0 ? -1.0f : 1.0f);

	hkReal xDir = (compressedXDir - toCompressedRangeX) / (toCompressedRangeX * almostOne);
	hkReal yDir = (compressedYDir - toCompressedRangeY) / (toCompressedRangeY * almostOne);
	hkReal zDir = compressedZDir * hkMath::sqrt(hkMath::max2(1 - xDir*xDir - yDir*yDir,0.0f));
#else
	hkReal compressedTheta = hkReal(accelerationDirection & 0xF);
	hkReal compressedPhi = hkReal(accelerationDirection >> 4);

	hkReal almostOne;
	almostOne = 1.0f - HK_REAL_EPSILON;
	hkReal toCompressedRange;
	toCompressedRange = 16.0f; //16 discrete steps total
	hkReal twoPi = 2.0f * HK_REAL_PI;

	hkReal xDir, yDir, zDir;
	hkReal theta = HK_REAL_PI * compressedTheta / (almostOne * toCompressedRange);
	hkReal phi = twoPi * compressedPhi / (almostOne * toCompressedRange);
	zDir = hkMath::cos(theta);
	xDir = yDir = hkMath::sin(theta);
	xDir *= hkMath::cos(phi);
	yDir *= hkMath::sin(phi);
#endif

	accelerationDirOut.set(xDir, yDir, zDir);
}

HK_FORCE_INLINE void hknpDeactivationStateUtil::accumAccelerationChange(hkSimdRealParameter accelerationDiff, hkInt8& accumAccelerationDiff)
{
	hkIntVector accumAccDiffVecI; accumAccDiffVecI.setAll(hkInt32(accumAccelerationDiff) + 128);//add 128 to go into unsigned integer space

	hkSimdReal scale; scale.setFromFloat(127.0f); //the maximum int8 value
	hkSimdReal offset; offset.setFromFloat(0.5f); //add 0.5 to account for truncation applied to toCompressedAccelDiff; desired behaviour is round to nearest integer
	hkSimdReal toCompressedAccDiff; toCompressedAccDiff.setAddMul(offset, accelerationDiff, scale);

	hkVector4 compressedAccDiffVec4; compressedAccDiffVec4.setAll(toCompressedAccDiff);
	hkIntVector compressedAccDiffVecI; compressedAccDiffVecI.setConvertF32toS32(compressedAccDiffVec4);
	hkIntVector newAccumAccDiffVecI; newAccumAccDiffVecI.setAddU32(accumAccDiffVecI, compressedAccDiffVecI);
	newAccumAccDiffVecI.setConvertU32ToU16(newAccumAccDiffVecI, newAccumAccDiffVecI);
	newAccumAccDiffVecI.setConvertSaturateS16ToU8(newAccumAccDiffVecI, newAccumAccDiffVecI);

	HK_ALIGN16(hkUint32 accumAcceleration32); newAccumAccDiffVecI.store<1>(&accumAcceleration32);
	accumAccelerationDiff = hkInt8((accumAcceleration32 & 0xFF) - 128);
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

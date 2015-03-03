/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

void hkQsTransformd::fastRenormalizeQuaternionBatch( hkQsTransformd* poseOut, hkUint32 numTransforms )
{

	// now normalize 4 quaternions at once
	hkQsTransformd* blockStart = poseOut;
	hkUint32 numTransformsOver4 = numTransforms/4;
	for (hkUint32 i=0; i< numTransformsOver4; i++)
	{
		hkVector4d dots;
		hkVector4dUtil::dot4_4vs4(blockStart[0].m_rotation.m_vec, blockStart[0].m_rotation.m_vec,
			blockStart[1].m_rotation.m_vec, blockStart[1].m_rotation.m_vec,
			blockStart[2].m_rotation.m_vec, blockStart[2].m_rotation.m_vec,
			blockStart[3].m_rotation.m_vec, blockStart[3].m_rotation.m_vec,
			dots);
		hkVector4d inverseSqrtDots;
		inverseSqrtDots.setSqrtInverse(dots);

		blockStart[0].m_rotation.m_vec.mul(inverseSqrtDots.getComponent<0>());
		blockStart[1].m_rotation.m_vec.mul(inverseSqrtDots.getComponent<1>());
		blockStart[2].m_rotation.m_vec.mul(inverseSqrtDots.getComponent<2>());
		blockStart[3].m_rotation.m_vec.mul(inverseSqrtDots.getComponent<3>());

		blockStart += 4;
	}

	hkUint32 leftovers = numTransforms%4;
	for (hkUint32 j=0; j<leftovers; j++)
	{
		blockStart[j].m_rotation.normalize();
	}
}

void hkQsTransformd::fastRenormalizeBatch( hkQsTransformd* poseOut, hkDouble64* weight, hkUint32 numTransforms)
{
	for (hkUint32 i=0; i < numTransforms; i++)
	{
		hkSimdDouble64 sweight; sweight.setFromFloat(weight[i]);
		hkSimdDouble64 invWeight; invWeight.setReciprocal(sweight);
		poseOut[i].m_translation.mul(invWeight);
		poseOut[i].m_scale.mul(invWeight);
	}

	fastRenormalizeQuaternionBatch(poseOut, numTransforms);
}

void hkQsTransformd::fastRenormalizeBatch( hkQsTransformd* poseOut, hkDouble64 weight, hkUint32 numTransforms)
{
	hkSimdDouble64 sweight; sweight.setFromFloat(weight);
	hkSimdDouble64 invWeight; invWeight.setReciprocal(sweight);

	for (hkUint32 i=0; i < numTransforms; i++)
	{	
		poseOut[i].m_translation.mul(invWeight);
		poseOut[i].m_scale.mul(invWeight);
	}

	fastRenormalizeQuaternionBatch(poseOut, numTransforms);
}

void hkQsTransformd::get4x4ColumnMajor(hkDouble64* HK_RESTRICT d) const
{

	// Calculate the 3x3 matrices for rotation and scale
	hkRotationd rotMatrix; rotMatrix.set (m_rotation);
	hkRotationd scaMatrix;
	hkMatrix3dUtil::_setDiagonal(m_scale, scaMatrix);

	// Calculate R*S
	hkRotationd rotSca;
	rotSca.setMul(rotMatrix, scaMatrix);

	// Construct hkTransformd
	hkTransformd temp; temp.set(rotSca, m_translation);

	// Use hkTransformd implementation
	temp.get4x4ColumnMajor(d);

}

bool hkQsTransformd::set4x4ColumnMajor(const hkDouble64* p)
{
	hkMatrixdDecomposition::Decomposition decomposition;
	hkMatrixdDecomposition::decompose4x4ColTransform(p, decomposition);

	set(decomposition.m_translation, decomposition.m_rotation, decomposition.m_scale);

	return !decomposition.m_hasSkew;
}

bool hkQsTransformd::set(const hkMatrix4d& m)
{
	hkMatrixdDecomposition::Decomposition decomposition;
	hkMatrixdDecomposition::decomposeMatrix(m, decomposition);

	set(decomposition.m_translation, decomposition.m_rotation, decomposition.m_scale);

	return !decomposition.m_hasSkew;
}

void hkQsTransformd::setFromTransformNoScale (const hkTransformd& transform)
{
	m_rotation.set(transform.getRotation());
	m_translation = transform.getTranslation();
	m_scale = hkVector4d::getConstant<HK_QUADREAL_1>();
}

void hkQsTransformd::copyToTransformNoScale (hkTransformd& transformOut) const
{
	transformOut.set(m_rotation, m_translation);
}

void hkQsTransformd::setFromTransform (const hkTransformd& transform)
{
	set4x4ColumnMajor(&transform(0,0));
}

//
//	Conversion from hkQTransformd

void hkQsTransformd::setFromTransform(const hkQTransformd& qt)
{
	m_rotation = qt.getRotation();
	m_translation = qt.getTranslation();
	m_scale = hkVector4d::getConstant<HK_QUADREAL_1>();
}

void hkQsTransformd::copyToTransform (hkTransformd& transformOut) const
{
	get4x4ColumnMajor( &transformOut(0,0) );
}


bool hkQsTransformd::isOk(const hkDouble64 epsilon) const
{
	bool transOk = m_translation.isOk<3>();
	bool scaleOk = m_scale.isOk<3>();
	bool rotOk   = m_rotation.isOk(epsilon);
	return transOk && scaleOk && rotOk;
}

bool hkQsTransformd::isApproximatelyEqual( const hkQsTransformd& other, hkDouble64 epsilon ) const
{
	hkSimdDouble64 sEps; sEps.setFromFloat(epsilon);
	// Make sure they are both in same hemisphere
	hkQuaterniond closestOther;
	closestOther.setClosest(other.m_rotation, m_rotation);

	return m_rotation.m_vec.allEqual<4>(closestOther.m_vec, sEps)  && m_translation.allEqual<3>( other.m_translation, sEps ) && m_scale.allEqual<3>(other.m_scale, sEps);
}


// Global instance used by hkQsTransformd::getIdentity()
HK_ALIGN_DOUBLE( hkDouble64 hkQsTransformd_identityStorage[12] ) =
{
	0,0,0,0, // position
	0,0,0,1, // rotation
	1,1,1,0, // scale  // do we need the 0 here?
};

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

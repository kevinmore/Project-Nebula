/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Base/KeyCode.h>

// Registration function is at the end of the file

static void hkMotionState_2_to_3(hkDataObject& obj)
{
	const hkDataObject src = obj["sweptTransform_old"].asObject();
	hkDataArray dst = obj["sweptTransform"].asArray();

	dst[0] = src["centerOfMass0"].asVector4();
	dst[1] = src["centerOfMass1"].asVector4();
	dst[2] = src["rotation0"].asVector4();
	dst[3] = src["rotation1"].asVector4();
	dst[4] = src["centerOfMassLocal"].asVector4();
}

static void hkSkinnedRefMeshShape_0_to_1(hkDataObject& obj)
{
	hkDataArray src = obj["localFromRootTransforms_old"].asArray(); // array of hkQTransform
	hkDataArray dst = obj["localFromRootTransforms"].asArray();     // array of 2xhkVector4

	const int size = src.getSize();
	dst.setSize(size*2);

	for (int i = 0; i < size; i++)
	{
		hkDataObject qt = src[i].asObject(); // one hkQTransform

		dst[i*2  ] = qt["rotation"].asVector4();
		dst[i*2+1] = qt["translation"].asVector4();
	}
}

static void hkxVertexBufferVertexData_0_to_1(hkDataObject& obj)
{
	hkDataArray src = obj["old_vectorData"].asArray();
	hkDataArray dst = obj["vectorData"].asArray();

	const int size = src.getSize();
	dst.setSize(size*4);
	for (int i = 0; i < size; i++)
	{
		const hkVector4& vec = src[i].asVector4();

		dst[i*4  ] = hkFloat32(vec(0));
		dst[i*4+1] = hkFloat32(vec(1));
		dst[i*4+2] = hkFloat32(vec(2));
		dst[i*4+3] = hkFloat32(vec(3));
	}
}

static void hkxVertexVectorDataChannel_1_to_2(hkDataObject& obj)
{
	hkDataArray src = obj["old_perVertexVectors"].asArray();
	hkDataArray dst = obj["perVertexVectors"].asArray();

	const int size = src.getSize();
	dst.setSize(size*4);
	for (int i = 0; i < size; i++)
	{
		const hkVector4& vec = src[i].asVector4();

		dst[i*4  ] = hkFloat32(vec(0));
		dst[i*4+1] = hkFloat32(vec(1));
		dst[i*4+2] = hkFloat32(vec(2));
		dst[i*4+3] = hkFloat32(vec(3));
	}
}

static void hkxAnimatedVector_1_to_2(hkDataObject& obj)
{
	hkDataArray src = obj["old_vectors"].asArray();
	hkDataArray dst = obj["vectors"].asArray();

	const int size = src.getSize();
	dst.setSize(size*4);
	for (int i = 0; i < size; i++)
	{
		const hkVector4& vec = src[i].asVector4();

		dst[i*4  ] = hkFloat32(vec(0));
		dst[i*4+1] = hkFloat32(vec(1));
		dst[i*4+2] = hkFloat32(vec(2));
		dst[i*4+3] = hkFloat32(vec(3));
	}
}

static void hkxAnimatedQuaternion_1_to_2(hkDataObject& obj)
{
	hkDataArray src = obj["old_quaternions"].asArray();
	hkDataArray dst = obj["quaternions"].asArray();

	const int size = src.getSize();
	dst.setSize(size*4);
	for (int i = 0; i < size; i++)
	{
		const hkQuaternion& vec = src[i].asQuaternion();

		dst[i*4  ] = hkFloat32(vec.m_vec(0));
		dst[i*4+1] = hkFloat32(vec.m_vec(1));
		dst[i*4+2] = hkFloat32(vec.m_vec(2));
		dst[i*4+3] = hkFloat32(vec.m_vec(3));
	}
}

static void hkxAnimatedMatrix_1_to_2(hkDataObject& obj)
{
	hkDataArray src = obj["old_matrices"].asArray();
	hkDataArray dst = obj["matrices"].asArray();

	const int size = src.getSize();
	dst.setSize(size*16);
	for (int i = 0; i < size; i++)
	{
		const hkMatrix4& m = src[i].asMatrix4();

		{
			const hkVector4& vec = m.getColumn<0>();
			dst[i*16  ] = hkFloat32(vec(0));
			dst[i*16+1] = hkFloat32(vec(1));
			dst[i*16+2] = hkFloat32(vec(2));
			dst[i*16+3] = hkFloat32(vec(3));
		}
		{
			const hkVector4& vec = m.getColumn<1>();
			dst[i*16+4] = hkFloat32(vec(0));
			dst[i*16+5] = hkFloat32(vec(1));
			dst[i*16+6] = hkFloat32(vec(2));
			dst[i*16+7] = hkFloat32(vec(3));
		}
		{
			const hkVector4& vec = m.getColumn<2>();
			dst[i*16+8] = hkFloat32(vec(0));
			dst[i*16+9] = hkFloat32(vec(1));
			dst[i*16+10] = hkFloat32(vec(2));
			dst[i*16+11] = hkFloat32(vec(3));
		}
		{
			const hkVector4& vec = m.getColumn<3>();
			dst[i*16+12] = hkFloat32(vec(0));
			dst[i*16+13] = hkFloat32(vec(1));
			dst[i*16+14] = hkFloat32(vec(2));
			dst[i*16+15] = hkFloat32(vec(3));
		}
	}
}

static void hkSkinnedMeshShapePart_0_to_1(hkDataObject& obj)
{
	const int boneIdx	= obj["boneIndex"].asInt();
	obj["boneSetId"]	= boneIdx;
}

static void hkStorageSkinnedMeshShape_0_to_1(hkDataObject& obj)
{
	hkDataArray boneSections	= obj["boneSections"].asArray();
	const int numBoneSections	= boneSections.getSize();

	int numTotalBones = 0;
	for (int si = numBoneSections - 1; si >= 0; si--)
	{
		hkDataObject section	= boneSections[si].asObject();
		const int boneIdx		= section[section.hasMember("startBoneIndex") ? "startBoneIndex" : "startBoneSetId"].asInt();
		const int numBones		= section[section.hasMember("numBones") ? "numBones" : "numBoneSets"].asInt();
		const int maxNumBones	= boneIdx + numBones;

		numTotalBones = (numTotalBones < maxNumBones) ? maxNumBones : numTotalBones;
	}

	hkDataArray bonesBuffer = obj["bonesBuffer"].asArray();
	hkDataArray boneSets	= obj["boneSets"].asArray();
	bonesBuffer.setSize(numTotalBones);
	boneSets.setSize(numTotalBones);

	for (int bi = 0; bi < numTotalBones; bi++)
	{
		bonesBuffer[bi] = bi;

		hkDataObject boneSet = boneSets[bi].asObject();
		boneSet["boneBufferOffset"]	= bi;
		boneSet["numBones"]			= 1;
		boneSets[bi] = boneSet;
	}
	
	obj["bonesBuffer"]	= bonesBuffer;
	obj["boneSets"]		= boneSets;
}

static void hkSkinnedMeshShapeBoneSection_0_to_1(hkDataObject& obj)
{
	const int boneIdx	= obj["startBoneIndex"].asInt();
	const int numBones	= obj["numBones"].asInt();
	obj["startBoneSetId"]	= boneIdx;
	obj["numBoneSets"]		= numBones;
}

void HK_CALL registerCommonPatches_2012_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2012_2/hkPatches_2012_2.cxx>
#	include <Common/Serialize/Version/hkVersionPatchManager.cxx>
#	undef HK_PATCHES_FILE
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

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

static void hclClothStateBufferAccess_0_to_1(hkDataObject& obj)
{
	obj["shadowBufferIndex"] = obj["bufferIndex"].asInt();
}

static void hclSceneDataSetupMesh_0_to_1(hkDataObject& obj)
{
	const hkDataArray oldArray = obj["old_meshBufferInterfaces"].asArray();

	hkDataArray newArray = obj["meshBufferInterfaces"].asArray();

	newArray.setSize(oldArray.getSize());
	for (int i=0; i<oldArray.getSize(); ++i)
	{
		newArray[i] = oldArray[i].asObject();
	}
}

// Old enum
enum
{
	HCL_NO_ACCESS = 0,
	HCL_POSITION_ACCESS_READ_ONLY = 1,
	HCL_POSITION_ACCESS_WRITE_ONLY = 2,
	HCL_POSITION_ACCESS_WRITE_ALL = 4,
	HCL_POSITION_ACCESS_READ_WRITE = (HCL_POSITION_ACCESS_READ_ONLY | HCL_POSITION_ACCESS_WRITE_ONLY), // 3, Slowest
	HCL_POSITION_ACCESS_MASK = (HCL_POSITION_ACCESS_READ_WRITE | HCL_POSITION_ACCESS_WRITE_ALL),
};

// New per-component enum
enum
{
	USAGE_NONE = 0,
	USAGE_READ = 1,
	USAGE_WRITE = 2,
	USAGE_FULL_WRITE = 4,
	USAGE_READ_BEFORE_WRITE = 8,
};

static void hclClothStateBufferAccess_1_to_2(hkDataObject& obj)
{
	int accessFlags = obj["accessFlags"].asInt();

	hkDataObject newUsage = obj["bufferUsage"].asObject();

	// 4 components
	for (int i=0; i<4; i++)
	{
		// Unfortunately, bitangents skipped bit 9!
		const int shift = (i==3) ? 10 : (i*3);
		int oldComponentFlags = (accessFlags >> shift) & HCL_POSITION_ACCESS_MASK;

		hkUint8 newComponentFlags = 0;
		if (oldComponentFlags & HCL_POSITION_ACCESS_READ_ONLY)
		{
			newComponentFlags |= USAGE_READ;
		}
		if (oldComponentFlags & HCL_POSITION_ACCESS_WRITE_ONLY)
		{
			newComponentFlags |= USAGE_WRITE;
		}
		if (oldComponentFlags & HCL_POSITION_ACCESS_READ_WRITE)
		{
			newComponentFlags |= USAGE_READ_BEFORE_WRITE; // Assume the worst
		}
		if (oldComponentFlags & HCL_POSITION_ACCESS_WRITE_ALL)
		{
			newComponentFlags |= USAGE_FULL_WRITE;
		}

		newUsage["perComponentFlags"].asArray()[i] = newComponentFlags;

	}

	newUsage["trianglesRead"] = false;
}

static void hclSkinOperator_1_to_2(hkDataObject& obj)
{
	const int numVerts = obj["endVertex"].asInt() - obj["startVertex"].asInt();

	hkDataArray influences = obj["boneInfluenceStartPerVertex"].asArray();

	bool partial = false;
	for (int i=0; i<numVerts; ++i)
	{
		const int startEntry = influences[i].asInt();
		const int endEntry = influences[i+1].asInt();
		const int numEntries = endEntry-startEntry;

		if (numEntries==0)
		{
			partial =true;
			break;
		}
	}

	obj["partialSkinning"] = partial;
}

static void hclGatherAllVerticesOperator_0_to_1(hkDataObject& obj)
{
	hkDataArray viFromVo = obj["vertexInputFromVertexOutput"].asArray();

	bool partial = false;
	for (int i=0; i<viFromVo.getSize(); ++i)
	{
		const int vi = viFromVo[i].asInt();

		if (vi<0)
		{
			partial =true;
			break;
		}
	}

	obj["partialGather"] = partial;
}

static void hclMeshMeshDeformOperator_1_to_2(hkDataObject& obj)
{
	const int numVerts = obj["endVertex"].asInt() - obj["startVertex"].asInt();

	hkDataArray influences = obj["triangleVertexStartForVertex"].asArray();

	bool partial = false;
	for (int i=0; i<numVerts; ++i)
	{
		const int startEntry = influences[i].asInt();
		const int endEntry = influences[i+1].asInt();
		const int numEntries = endEntry-startEntry;

		if (numEntries==0)
		{
			partial =true;
			break;
		}
	}

	obj["partialDeform"] = partial;
}

void HK_CALL registerClothPatches_660(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/660/hclPatches_660.cxx>
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

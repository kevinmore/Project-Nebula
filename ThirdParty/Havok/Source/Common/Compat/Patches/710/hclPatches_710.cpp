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

static void hclSimClothData_0_to_1(hkDataObject& obj)
{
	const int numParticles = obj["particleDatas"].asArray().getSize();

	hkDataArray masks = obj["staticCollisionMasks"].asArray();
	masks.setSize(numParticles);
	for (int i=0; i<numParticles; ++i)
	{
		masks[i] = 0;
	}

	hkDataArray pairs = obj["staticCollisionPairs"].asArray();

	const int numPairs = pairs.getSize();

	for (int i=0; i<numPairs; ++i)
	{
		const int collidable = pairs[i].asObject()["collidableIndex"].asInt();
		const int particle = pairs[i].asObject()["particleIndex"].asInt();

		masks[particle] = masks[particle].asInt() | 1<<collidable;
	}

	obj["maxCollisionPairs"] = numPairs;
}

static void hclCapsuleShape_0_to_1(hkDataObject& obj)
{
	const hkReal capLenSqrd = obj["capLenSqrd"].asReal();
	obj["capLenSqrdInv"]=1.0f/capLenSqrd;
}

static void hclSimClothData_1_to_2(hkDataObject& obj)
{
	hkDataObject newStruct = obj["collidableTransformMap"].asObject();
	newStruct["transformSetIndex"]=-1;
}

static void hclSimClothData_2_to_3(hkDataObject& obj)
{
	// Determine whether old sim cloth data requested a global normal flip or not
	hkDataObject simulationInfo = obj["simulationInfo"].asObject();
	bool doNormals = ( simulationInfo["doNormals"].asInt() != 0 );

	if (doNormals)
	{
		// And set the per-triangle flips all to the same value accordingly
		const int numTriangles = obj["triangleIndices"].asArray().getSize()/3;

		// Make array containing a bit per triangle flip
		hkUint32 numBytes = numTriangles/8;
		if (numTriangles % 8) numBytes++;

		hkUint32 numInt32 = numBytes/4;
		if (numBytes % 4) numInt32++;

		hkDataArray triangleFlips = obj["triangleFlips"].asArray();

		bool flipNormals = ( simulationInfo["flipNormals"].asInt() != 0 );
		hkUint32 defaultFlip = flipNormals ? 0xffffffff : 0;

		triangleFlips.setSize(numInt32);
		for (hkUint32 i=0; i<numInt32; ++i)
		{
			triangleFlips[i] = (int)defaultFlip;
		}
	}
}

static void hclUpdateAllVertexFramesOperator_1_to_2(hkDataObject& obj)
{
	bool flipNormals = ( obj["flipNormals"].asInt() != 0 );

	// The old operator doesn't store the total number of triangles. We have to just over-estimate it here.
	const int numTriangles = 3 * obj["referenceVertices"].asArray().getSize();

	// Make array containing a bit per triangle flip
	hkUint32 numBytes = numTriangles/8;
	if (numTriangles % 8) numBytes++;

	hkUint32 numInt32 = numBytes/4;
	if (numBytes % 4) numInt32++;

	hkDataArray triangleFlips = obj["triangleFlips"].asArray();

	hkUint32 defaultFlip = flipNormals ? 0xffffffff : 0;

	triangleFlips.setSize(numInt32);
	for (hkUint32 i=0; i<numInt32; ++i)
	{
		triangleFlips[i] = (int)defaultFlip;
	}
}

static void hclUpdateSomeVertexFramesOperator_1_to_2(hkDataObject& obj)
{
	const int numTriangles = obj["involvedTriangles"].asArray().getSize();

	// Make array containing a bit per triangle flip
	hkUint32 numBytes = numTriangles/8;
	if (numTriangles % 8) numBytes++;

	hkUint32 numInt32 = numBytes/4;
	if (numBytes % 4) numInt32++;

	hkDataArray triangleFlips = obj["triangleFlips"].asArray();

	bool flipNormals = ( obj["flipNormals"].asInt() != 0 );

	hkUint32 defaultFlip = flipNormals ? 0xffffffff : 0;
	triangleFlips.setSize(numInt32);

	for (hkUint32 i=0; i<numInt32; ++i)
	{
		triangleFlips[i] = (int)defaultFlip;
	}
}

static void hclVolumeConstraintSetupObject_0_to_1(hkDataObject& obj)
{
	obj["useDeprecatedMethod"] = 1;
}

static void hclVolumeConstraint_0_to_1(hkDataObject& obj)
{
	obj["useDeprecatedMethod"] = 1;
}

static void hclSimClothData_3_to_4(hkDataObject& obj)
{
	hkReal maxRadius = 0.0f;
	hkDataArray particleDatas = obj["particleDatas"].asArray();
	const int numParticles = particleDatas.getSize();

	for (int i=0; i<numParticles; ++i)
	{
		const hkReal radius = particleDatas[i].asObject()["radius"].asReal();
		if (radius>maxRadius) maxRadius = radius;
	}

	obj["maxParticleRadius"] = maxRadius;
}

static void hclLocalRangeConstraintSet_0_to_1(hkDataObject& obj)
{
	bool normalComponentUsed = false;
	hkDataArray entries = obj["localConstraints"].asArray();
	const int numEntries = entries.getSize();

	for (int i=0; i<numEntries; ++i)
	{
		const hkDataObject entry = entries[i].asObject();
		const hkReal maxDist = entry["maximumDistance"].asReal();
		const hkReal maxNDist = entry["maxNormalDistance"].asReal();
		const hkReal minNDist = entry["minNormalDistance"].asReal();

		normalComponentUsed = normalComponentUsed || (maxNDist<maxDist);
		normalComponentUsed = normalComponentUsed || (minNDist>(-maxDist));
	}

	obj["applyNormalComponent"] = normalComponentUsed;
}

void HK_CALL registerClothPatches_710(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/710/hclPatches_710.cxx>
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

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

static void hkpConstraintInstance_0_to_1(hkDataObject& obj)
{
	// see HVK-4668, HVK-4669
	// enum ConstraintPriority changed - value added in the middle
	int curEnum = obj["priority"].asInt();
	if( curEnum >= 2 ) // starting hkpConstraintInstance::PRIORITY_TOI
	{
		// should increase the current value
		obj["priority"] = ++curEnum;
	}
}

static void CopyUint32ToMaterial(hkDataArray intArray, hkDataArray materialArray)
{
	if( intArray.getSize() == 0 )
	{
		return;
	}
	HK_ASSERT(0x650fe9d6, materialArray.getSize() == 0);
	materialArray.setSize(intArray.getSize());
	hkDataClass materialClass = materialArray.getClass();
	HK_ASSERT(0x35d75156, !materialClass.isNull());
	const hkDataWorld* world = materialClass.getWorld();

	// workaround for hkHalf type
	hkDataClass halfClass = world->findClass("hkHalf");
	HK_ASSERT(0x3bcde0ce, !halfClass.isNull());
	hkHalf one; one = 1.0f;
	hkInt16 oneInt16 = *reinterpret_cast<hkInt16*>(&one);

	for( int i = 0; i < intArray.getSize(); ++i )
	{
		hkDataObject material = world->newObject(materialClass);
		// the struct must have all members initialized (including zeros)
		// before it added into array
		material["filterInfo"] = intArray[i].asInt();
		hkDataObject restitution = world->newObject(halfClass);
		restitution["value"] = 0;
		material["restitution"] = restitution;
		hkDataObject friction = world->newObject(halfClass);
		friction["value"] = oneInt16;
		material["friction"] = friction;
		material["userData"] = 0;
		materialArray[i] = material;
	}
}

static void hkpStorageExtendedMeshShapeMeshSubpartStorage_0_to_1(hkDataObject& obj)
{
	CopyUint32ToMaterial(obj["int_materials"].asArray(), obj["materials"].asArray());
}

static void hkpStorageExtendedMeshShapeShapeSubpartStorage_0_to_1(hkDataObject& obj)
{
	CopyUint32ToMaterial(obj["int_materials"].asArray(), obj["materials"].asArray());
}

static void CopyObject(hkDataObject& obj, const char* oldMember, const char* newMember)
{
	obj[newMember] = obj[oldMember].asObject();
}

static void hkpMotion_0_to_1(hkDataObject& obj)
{
	CopyObject(obj, "max_savedMotion", "savedMotion");
}

static void hkpEntity_0_to_1(hkDataObject& obj)
{
	CopyObject(obj, "max_motion", "motion");
}

static void hkpVehicleInstanceWheelInfo_0_to_1(hkDataObject& obj)
{
	obj["contactShapeKey"].asArray()[0] = obj["old_contactShapeKey"].asInt();
	for (int i=1; i<8; ++i)
	{
		obj["contactShapeKey"].asArray()[i] = -1;
	}
}

static void hkpMassChangerModifierConstraintAtom_0_to_1(hkDataObject& obj)
{
	obj["factorA"] = obj["old_factorA"].asReal();
	obj["factorB"] = obj["old_factorB"].asReal();
}

static void hkpExtendedMeshShapeTrianglesSubpart_1_to_2(hkDataObject& obj)
{
	hkInt32 stridingType = obj["stridingType"].asInt();
	if( stridingType > 0 )
	{
		obj["stridingType"] = stridingType + 1;
	}
}

static inline void convertStructToHalf(hkDataObject& obj, const char* structMemberName, const char* halfMemberName)
{
	hkDataObject halfStruct = obj[structMemberName].asObject();
#if defined(HK_HALF_IS_FLOAT)
	HK_COMPILE_TIME_ASSERT(hkSizeOf(hkHalf) == hkSizeOf(hkFloat32));
	hkFloat32 value = halfStruct.isNull() ? 0.0f : hkFloat32(halfStruct["value"].asReal());
#else
	HK_COMPILE_TIME_ASSERT(hkSizeOf(hkHalf) == hkSizeOf(hkInt16));
	hkInt16 value = halfStruct.isNull() ? 0 : hkInt16(halfStruct["value"].asInt());
#endif
	hkHalf half = *reinterpret_cast<hkHalf*>(&value);
	obj[halfMemberName] = half;
}

static void hkpStorageExtendedMeshShapeMaterial_0_to_1(hkDataObject& obj)
{
	convertStructToHalf(obj, "old_restitution", "restitution");
	convertStructToHalf(obj, "old_friction", "friction");
}

static void hkpMotion_1_to_2(hkDataObject& obj)
{
	convertStructToHalf(obj, "old_gravityFactor", "gravityFactor");
}

static void hkpMotion_2_to_3(hkDataObject& obj)
{
	// Motion Type enumeration changed. Stabilized -box and -sphere motion types removed.

	// Types were:
	// MOTION_SPHERE_INERTIA = 2
	// MOTION_STABLILIZED_SPHERE_INERTIA = 3 // gone now
	// MOTION_BOX_INERTIA = 4
	// MOTION_STABILIZED_BOX_INERTIA = 5 // gone now

	int motionType = obj["type"].asInt();

	if (motionType > 2)
	{
		if (motionType > 4)
		{
			// replace stablized box with normal box & shift higher vals.
			motionType--;
		}

		// replace stablized sphere with normal sphere & shift higher vals.
		motionType--;

		// write back
		obj["type"] = motionType;
	}
}

void HK_CALL registerPhysicsPatches_Legacy(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/Legacy/hkpPatches_Legacy.cxx>
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

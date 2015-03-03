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

static void hkdBreakableShape_5_to_6(hkDataObject& obj)
{
	hkDataObject inertias = obj["inertiaAndValues"].asObject();
	for (int i=0; i<8; ++i)
	{
		const hkUint16 tmp = (hkUint16)inertias["halfs"].asArray()[i].asInt();
		const hkHalf h = *(const hkHalf*)&tmp;
		const hkReal r = (hkReal)h;
		obj["temp_inertiaAndValues"].asArray()[i] = r;
	}
	// Save current physics shape
	hkDataObject physicsShape = obj["oldPhysicsShape"].asObject();
	obj["physicsShape"] = physicsShape;

	// Create a simple material and set it on the shape
	const hkDataWorld* world = obj.getClass().getWorld();
	hkDataClass simpleMaterialClass(world->findClass("hkpSimpleBreakableMaterial"));
	hkDataObject simpleMaterial = world->newObject(simpleMaterialClass);
	simpleMaterial["strength"] = obj["strength"];
	obj["material"] = simpleMaterial;
}

static void hkdDeformableBreakableShapePhysicsSkinInstance_0_to_1(hkDataObject& obj)
{
	// Create a new physics shape 
	const hkDataWorld* world = obj.getClass().getWorld();
	hkDataClass havokPhysicsShapeClass(world->findClass("hkpBreakableShape"));
	hkDataObject havokPhysicsShape = world->newObject(havokPhysicsShapeClass);

	havokPhysicsShape["physicsShape"] = obj["physicsShape"].asObject();
	obj["deformableCompound"] = havokPhysicsShape;
}

static void hkdBreakableBody_6_to_7(hkDataObject& obj)
{
	// Copy everything from child to parent class
	hkDataObject controller		= obj["oldController"].asObject();
	hkDataObject breakableShape	= obj["oldBreakableShape"].asObject();
	hkUint8 bodyAndFlags		= (hkUint8)obj["oldBodyTypeAndFlags"].asInt();
	hkReal constraintStrength	= (hkReal)obj["oldConstraintStrength"].asReal();

	obj["controller"]			= controller;
	obj["breakableShape"]		= breakableShape;
	obj["bodyTypeAndFlags"]		= bodyAndFlags;
	obj["constraintStrength"]	= constraintStrength;
}

static void hkdDeformableBreakableShapeBoneInfo_0_to_1(hkDataObject& obj)
{
	hkReal softness = obj["softness"].asReal();
	hkVector4 comAndSoftness;	comAndSoftness.set(0.0f, 0.0f, 0.0f, softness);
	obj["comAndSoftness"] = comAndSoftness;
}

void HK_CALL registerDestructionPatches_2011_1(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2011_1/hkdPatches_2011_1.cxx>
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

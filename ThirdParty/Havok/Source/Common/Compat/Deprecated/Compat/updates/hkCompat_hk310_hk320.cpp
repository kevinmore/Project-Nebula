/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Compat/Deprecated/Compat/hkCompatUtil.h>
#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>
#include <Common/Serialize/Version/hkVersionUtil.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

namespace hkCompat_hk310_hk320
{

static void ZeroPointer( hkVariant& newObj, const char* memName )
{
	hkClassMemberAccessor userData(newObj, memName);
	if( userData.isOk() )
	{
		userData.asPointer() = HK_NULL;
	}
}
static void ZeroUserDataPointer( hkVariant& newObj )
{
	ZeroPointer(newObj, "userData");
}

static void ZeroUserDataPointer(
	hkVariant& oldObj,
	hkVariant& newObj,
	hkObjectUpdateTracker& )
{
	ZeroUserDataPointer(newObj);
}

static void VehicleInstanceWheelInfo(
	hkVariant& oldObj,
	hkVariant& newObj,
	hkObjectUpdateTracker& )
{
	ZeroPointer( newObj, "contactBody" );
}

static void VehicleInstance_310_320(
									 hkVariant& oldObj,
									 hkVariant& newObj,
									 hkObjectUpdateTracker& tracker)
{
	hkClassMemberAccessor oldWheelsInfoArray(oldObj, "wheelsInfo");
	hkClassMemberAccessor newWheelsInfoArray(newObj, "wheelsInfo");

	hkClassMemberAccessor::SimpleArray& oldArray = oldWheelsInfoArray.asSimpleArray();
	hkClassMemberAccessor::SimpleArray& newArray = newWheelsInfoArray.asSimpleArray();
	HK_ASSERT( 0xad78555b, oldArray.size == newArray.size );
	hkVariant oldWheelInfoVariant = {HK_NULL, &oldWheelsInfoArray.object().getClass()};
	hkVariant newWheelInfoVariant = {HK_NULL, &newWheelsInfoArray.object().getClass()};
	hkInt32 oldWheelInfoSize = oldWheelInfoVariant.m_class->getObjectSize();
	hkInt32 newWheelInfoSize = newWheelInfoVariant.m_class->getObjectSize();
	for( int i = 0; i < oldArray.size; ++i )
	{
		oldWheelInfoVariant.m_object = static_cast<char*>(oldArray.data) + oldWheelInfoSize*i;
		newWheelInfoVariant.m_object = static_cast<char*>(newArray.data) + newWheelInfoSize*i;
		VehicleInstanceWheelInfo(oldWheelInfoVariant, newWheelInfoVariant, tracker);
	}
}

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// Physics

	// World object
	{ 0x0fd71e32, 0x7db00d3a, hkVersionRegistry::VERSION_INPLACE, "hkAabbPhantom", ZeroUserDataPointer }, // overlapping collidables zeroed
	{ 0xc0746727, 0x7a839e5e, hkVersionRegistry::VERSION_COPY, "hkEntity", ZeroUserDataPointer },	// new uid member added, userdata->zero
	{ 0xef940b7c, 0x4bcc8a37, hkVersionRegistry::VERSION_COPY, "hkWorldCinfo", HK_NULL }, // New members + defaults
	{ 0x0bdb4a79, 0x8102d162, hkVersionRegistry::VERSION_INPLACE, "hkWorldObject", ZeroUserDataPointer },  // broadphase border enum added, userdata->zero

	// Constraints
	{ 0xb1c5a2d7, 0xda7599af, hkVersionRegistry::VERSION_COPY, "hkBreakableConstraintData", HK_NULL }, // many additions
	{ 0xa487a781, 0x6553492d, hkVersionRegistry::VERSION_COPY, "hkLimitedHingeConstraintData", HK_NULL }, // angularLimitsTauFactor added
	{ 0xc1f5d50d, 0x22baaa3f, hkVersionRegistry::VERSION_COPY, "hkPoweredRagdollConstraintData", HK_NULL }, // copy as it actually got smaller and members moved (bool after matrix now instead of before)
	{ 0x09905d3a, 0x7ff383a1, hkVersionRegistry::VERSION_COPY, "hkRagdollConstraintData", HK_NULL }, // angular tau added
	
	// Misc
	{ 0x73004dbb, 0xed923a6e, hkVersionRegistry::VERSION_MANUAL, "hkContactPointMaterial", ZeroUserDataPointer }, // userdata -> zero
	{ 0xab4eb42f, 0xe15f41a4, hkVersionRegistry::VERSION_MANUAL, "hkPhysicsSystem", ZeroUserDataPointer },
	BINARY_IDENTICAL(0x51a73ef8, 0x1cd2a3e1, "hkMeshShapeSubpart"), // type of subpart members changed
		
	// Vehicle
	{ 0x26f7cf9f, 0xcd195550, hkVersionRegistry::VERSION_COPY, "hkVehicleInstance", VehicleInstance_310_320 }, // Changed friction Status, changed m_wheelsInfo
	{ 0xd8b13b98, 0x1c076a84, hkVersionRegistry::VERSION_COPY, "hkVehicleFrictionStatus", HK_NULL }, // Changed friction Status
	{ 0x5854d53d, 0xe70e2bb4, hkVersionRegistry::VERSION_COPY, "hkVehicleFrictionStatusAxisStatus", HK_NULL }, // Changed friction Status
	{ 0x5bf274c8, 0x8d5654ee, hkVersionRegistry::VERSION_MANUAL, "hkVehicleInstanceWheelInfo", VehicleInstanceWheelInfo }, // contactBody -> zero

	BINARY_IDENTICAL(0x1ff447b3, 0x9e4ee5d9, "hkClass"), // signature flags enum added
	BINARY_IDENTICAL(0x16df02fe, 0x8c11d1f2, "hkPackfileHeader"), // name added (size unchanged")
	BINARY_IDENTICAL(0xb33582ad, 0x953b00d8, "hkMotion"), // enum THIN_MOTION added
	BINARY_IDENTICAL(0x9eeaa6f5, 0xf5aa2dc6, "hkMeshShape"), // type of subpart members changed

	// Animation
	{ 0x4af7c559, 0xa35e6164, hkVersionRegistry::VERSION_COPY, "hkSkeleton", HK_NULL }, // added name
	{ 0xef964a74, 0xe39df839, hkVersionRegistry::VERSION_COPY, "hkAnimationBinding", HK_NULL }, // added blendHint

	// Scenedata
	{ 0x12a4e063, 0x12a4e063, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL }, // variant
	{ 0x35e1060e, 0x35e1060e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL }, // variant
	{ 0x9dd3289c, 0x9dd3289c, hkVersionRegistry::VERSION_VARIANT, "hkBoneAttachment", HK_NULL }, // variant
	{ 0xe085ba9f, 0xe085ba9f, hkVersionRegistry::VERSION_VARIANT, "hkxMaterialTextureStage", HK_NULL }, // variant
	{ 0x57061454, 0x57061454, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },
	{ 0x8b106e42, 0xd5c65fae, hkVersionRegistry::VERSION_COPY, "hkxCamera", HK_NULL }, // lefthanded added
	{ 0xba07287d, 0x2641039e, hkVersionRegistry::VERSION_COPY | hkVersionRegistry::VERSION_VARIANT, "hkxNode", HK_NULL }, // attribute groups added, variant
	{ 0xe0d8d8b2, 0x1c6f8636, hkVersionRegistry::VERSION_COPY, "hkxScene", HK_NULL }, // applied transform
	{ 0x9b712385, 0x3d43489c, hkVersionRegistry::VERSION_COPY | hkVersionRegistry::VERSION_VARIANT, "hkxMaterial", HK_NULL }, // added extra data variant (can store effect shader ptr etc)

	{ 0,0, 0, HK_NULL, HK_NULL }
};

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok310Classes
#define HK_COMPAT_VERSION_TO hkHavok320Classes
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO
} // namespace hkCompat_hk310_hk320

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

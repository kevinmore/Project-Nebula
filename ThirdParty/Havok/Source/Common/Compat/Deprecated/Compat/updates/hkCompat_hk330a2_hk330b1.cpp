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
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

namespace hkCompat_hk330a2_hk330b1
{

static void PositionConstraintMotor(
	hkVariant& oldObj,
	hkVariant& newObj,
	hkObjectUpdateTracker& )
{
	hkClassMemberAccessor newMin(newObj, "minForce");
	hkClassMemberAccessor newMax(newObj, "maxForce");
	hkClassMemberAccessor oldMax(oldObj, "maxForce");

	if (newMin.isOk() && newMax.isOk() && oldMax.isOk())
	{
		newMin.asReal() = - oldMax.asReal();
		newMax.asReal() = + oldMax.asReal();
	}
	else
	{
		HK_ASSERT2(0xad7d77de, false, "member not found");
	}
}

static void VelocityConstraintMotor(
	hkVariant& oldObj,
	hkVariant& newObj,
	hkObjectUpdateTracker& )
{
	hkVersionUtil::renameMember(oldObj, "maxNegForce", newObj, "minForce");
	hkVersionUtil::renameMember(oldObj, "maxPosForce", newObj, "maxForce");
}

static void SpringDamperConstraintMotor(
	hkVariant& oldObj,
	hkVariant& newObj,
	hkObjectUpdateTracker& )
{
	hkVersionUtil::renameMember(oldObj, "maxNegForce", newObj, "minForce");
	hkVersionUtil::renameMember(oldObj, "maxPosForce", newObj, "maxForce");
}

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }


static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	{ 0x1c50563b, 0xead954fd, hkVersionRegistry::VERSION_COPY, "hkPositionConstraintMotor", PositionConstraintMotor }, // max/min forces moved to base class
	{ 0xa03b1417, 0x94d2e665, hkVersionRegistry::VERSION_COPY, "hkVelocityConstraintMotor", VelocityConstraintMotor }, // member added, defaults to false + max/min forces moved to base class
	{ 0x48377d86, 0xb29a4f46, hkVersionRegistry::VERSION_COPY, "hkSpringDamperConstraintMotor", SpringDamperConstraintMotor }, // max/min forces moved to base class

	{0xbfc428d3, 0xbf0e8138, hkVersionRegistry::VERSION_COPY, "hkPoweredChainData", HK_NULL }, // Array<ConstraintInfo> changed
	{0x173a57ec, 0xf88aee25, hkVersionRegistry::VERSION_COPY, "hkPoweredChainDataConstraintInfo", HK_NULL }, // removed m_cfm*, moved m_bTc

	REMOVED("hkConstraintChainDriverDataTarget"),
	REMOVED("hkConstraintChainDriverData"),
	REMOVED("hkTriPatchTriangle"),

	BINARY_IDENTICAL(0xda8c7d7d, 0x82ef3c01, "hkConstraintMotor"), // enum added
	BINARY_IDENTICAL(0xb368f9bd, 0xde4be9fc, "hkConstraintData"), // enum type_chain_driver removed
	
	{ 0x9dd3289c, 0x9dd3289c, hkVersionRegistry::VERSION_VARIANT, "hkBoneAttachment", HK_NULL },
	{ 0x12a4e063, 0x12a4e063, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x35e1060e, 0x35e1060e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL },
	{ 0x3d43489c, 0x3d43489c, hkVersionRegistry::VERSION_VARIANT, "hkxMaterial", HK_NULL },
	{ 0xe085ba9f, 0xe085ba9f, hkVersionRegistry::VERSION_VARIANT, "hkxMaterialTextureStage", HK_NULL },
	{ 0x914da6c1, 0x914da6c1, hkVersionRegistry::VERSION_VARIANT, "hkxAttribute", HK_NULL },
	{ 0x8b69ead5, 0x8b69ead5, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeGroup", HK_NULL },
	{ 0xb926cec1, 0xb926cec1, hkVersionRegistry::VERSION_VARIANT, "hkxNode", HK_NULL },
	{ 0x57061454, 0x57061454, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

	BINARY_IDENTICAL(0xb330fa01, 0xff8ce40d, "hkPackfileHeader"), // contentsClass -> contentsClassName

	{ 0,0, 0, HK_NULL, HK_NULL }
};

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok330a2Classes
#define HK_COMPAT_VERSION_TO hkHavok330b1Classes
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO
} // namespace hkCompat_hk330a2_hk330b1

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

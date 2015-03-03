/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Compat/Deprecated/Compat/hkCompatUtil.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/Math/hkMath.h>
#include <Common/Base/Container/BitField/hkBitField.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

namespace hkCompat_hk460b2_hk460r1
{
	static void hkSerializedAgentNnEntry_hk460b2_hk460r1( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& )
	{
		HK_ASSERT2(0x54e32127, false, "Versioning is not implemented yet.");
	}

	static void hkSerializedContactPointPropertiesBlock_hk460b2_hk460r1( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& )
	{
		HK_ASSERT2(0x54e32128, false, "Versioning is not implemented yet.");
	}

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// common
	{ 0x8bdd3e9a, 0x8bdd3e9a, hkVersionRegistry::VERSION_VARIANT, "hkBoneAttachment", HK_NULL },
	{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL }, 
	{ 0xf2ec0c9c, 0xf2ec0c9c, hkVersionRegistry::VERSION_VARIANT, "hkxMaterial", HK_NULL },
	{ 0x914da6c1, 0x914da6c1, hkVersionRegistry::VERSION_VARIANT, "hkxAttribute", HK_NULL },
	{ 0x1667c01c, 0x1667c01c, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeGroup", HK_NULL }, 
	{ 0x06af1b5a, 0x06af1b5a, hkVersionRegistry::VERSION_VARIANT, "hkxNode", HK_NULL },
	{ 0xe085ba9f, 0xe085ba9f, hkVersionRegistry::VERSION_VARIANT, "hkxMaterialTextureStage", HK_NULL },
	{ 0x72e8e849, 0x72e8e849, hkVersionRegistry::VERSION_VARIANT, "hkxMesh", HK_NULL },
	{ 0x912c8863, 0x912c8863, hkVersionRegistry::VERSION_VARIANT, "hkxMeshSection", HK_NULL },
	{ 0x64e9a03c, 0x64e9a03c, hkVersionRegistry::VERSION_VARIANT, "hkxMeshUserChannelInfo", HK_NULL },
	{ 0x445a443a, 0x445a443a, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeHolder", HK_NULL },
	{ 0x57061454, 0x57061454, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

	// hkbehavior

	{ 0xb36b9af6, 0x563dba83, hkVersionRegistry::VERSION_COPY, "hkbBehavior", HK_NULL },
	{ 0x4d94add9, 0x20cc25f6, hkVersionRegistry::VERSION_COPY, "hkbStateMachineActiveTransitionInfo", HK_NULL },
	{ 0xc1f29013, 0xf61504a2, hkVersionRegistry::VERSION_COPY, "hkbStateMachine", HK_NULL },

	// hkconstraintsolver

	BINARY_IDENTICAL(0x4e7b027c, 0x86c62c9c, "hkMassChangerModifierConstraintAtom"), // fixed size padding (no save)

	// hkcollide

		// hkExtendedMeshShape: Complete 46r1 has 0x29ebe708 signature
	{ 0x7d392dbc, 0x97aa3ab6, hkVersionRegistry::VERSION_COPY, "hkExtendedMeshShape", HK_NULL }, // added m_embeddedTrianglesSubpart
	{ 0x4f54c5ac, 0xeb33369b, hkVersionRegistry::VERSION_COPY, "hkMoppBvTreeShape", HK_NULL }, // reorder members
	{ 0x15110a40, 0x5f31ebc7, hkVersionRegistry::VERSION_COPY, "hkSphereShape", HK_NULL }, // fixed size padding (no save)

	// hkutilities
	{ 0xbd2ac814, 0xcbeca93e, hkVersionRegistry::VERSION_COPY, "hkSerializedAgentNnEntry", hkSerializedAgentNnEntry_hk460b2_hk460r1 },
	{ 0xd699c965, 0xfaa46bcc, hkVersionRegistry::VERSION_COPY, "hkSerializedContactPointPropertiesBlock", hkSerializedContactPointPropertiesBlock_hk460b2_hk460r1 },

	{ 0, 0, 0, HK_NULL, HK_NULL }
};

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok460b2Classes
#define HK_COMPAT_VERSION_TO hkHavok460r1Classes
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO
} // namespace hkCompat_hk460b2_hk460r1

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

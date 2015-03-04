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
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/hkBase.h>
#include <Common/Base/Container/BitField/hkBitField.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

namespace hkHavok460r1Classes
{
	extern hkClass hkExtendedMeshShapeClass;
	extern hkClass hkSphereShapeClass;
	extern hkClass hkMassChangerModifierConstraintAtomClass;
}

namespace hkCompat_hk460r1_hk461r1
{
	// This function searches and updates all found meta data (hkClass) for
	// three classes as they were modified between Complete and
	// Spectrum 460r1 releases:
	//  - hkExtendedMeshShape
	//  - hkSphereShape
	//  - hkMassChangerModifierConstraintAtom
	// NOTE: The 460r1 class manifest reflects meta data for Spectrum 460r1 release.
	//
	static void UpdateMetadataFor460r1CompleteObjects(hkArray<hkVariant>& objectsInOut)
	{
		for( int i=0; i < objectsInOut.getSize(); ++i )
		{
			const hkClass* klass = objectsInOut[i].m_class;
			HK_ASSERT(0x54e32125, klass);
			const char* klassName = klass->getName();
			if( hkString::strCmp(klassName, "hkExtendedMeshShape") == 0 )
			{
				objectsInOut[i].m_class = &hkHavok460r1Classes::hkExtendedMeshShapeClass;
				HK_ASSERT2(0x54e32126, klass->getSignature() == objectsInOut[i].m_class->getSignature(), "The hkExtendedMeshShape object is corrupt and no versioning can be done.");
			}
			else if( hkString::strCmp(klassName, "hkSphereShape") == 0 )
			{
				objectsInOut[i].m_class = &hkHavok460r1Classes::hkSphereShapeClass;
			}
			else if( hkString::strCmp(klassName, "hkMassChangerModifierConstraintAtom") == 0 )
			{
				objectsInOut[i].m_class = &hkHavok460r1Classes::hkMassChangerModifierConstraintAtomClass;
			}
		}
	}
	static void Nop(hkVariant&, hkVariant&, hkObjectUpdateTracker&) {}

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

	// physics
	{ 0x5f31ebc7, 0x795d9fa, hkVersionRegistry::VERSION_COPY, "hkSphereShape", Nop }, // Fixed padding for dma, HVK-4027

	// behavior
	{ 0x822a7bef, 0x6f02f92a, hkVersionRegistry::VERSION_COPY, "hkbBlendingTransitionEffect", HK_NULL }, // new members
	{ 0xf61504a2, 0xb5cd4e89, hkVersionRegistry::VERSION_COPY, "hkbStateMachine", HK_NULL }, // new members
	{ 0x9dca84ac, 0xfea091e8, hkVersionRegistry::VERSION_COPY, "hkbStateMachineStateInfo", HK_NULL }, // new members
	{ 0xbdcda8e5, 0x35f9d035, hkVersionRegistry::VERSION_COPY, "hkbStateMachineTransitionInfo", HK_NULL }, // new members

	{ 0, 0, 0, HK_NULL, HK_NULL }
};

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok460r1Classes
#define HK_COMPAT_VERSION_TO hkHavok461r1Classes
#define HK_COMPAT_OPTIONAL_UPDATE_FUNC update

extern hkVersionRegistry::UpdateDescription hkVersionUpdateDescription;

static hkResult HK_CALL update(
	hkArray<hkVariant>& objectsInOut,
	hkObjectUpdateTracker& tracker )
{
	UpdateMetadataFor460r1CompleteObjects(objectsInOut);
	return hkVersionUtil::updateSingleVersion( objectsInOut, tracker, hkVersionUpdateDescription, hkVersionRegistry::getInstance().getClassNameRegistry(HK_COMPAT_VERSION_TO::VersionString) );
}

#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO
} // namespace hkCompat_hk460r1_hk461r1

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

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
#include <Common/Base/hkBase.h>
#include <Common/Base/Container/BitField/hkBitField.h>

#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

namespace hkCompat_hk550b2_hk550r1
{
	static void update_hkbFootIkGains( hkVariant& oldObj, hkVariant& newObj )
	{
		// a gain was renamed
		{
			hkClassMemberAccessor oldMember(oldObj, "pelvisFeedbackGain");
			hkClassMemberAccessor newMember(newObj, "worldFromModelFeedbackGain");

			newMember.asReal() = oldMember.asReal();
		}
	}

	static void update_hkbFootIkControlData( hkVariant& oldObj, hkVariant& newObj )
	{
		// version the gains
		{
			hkClassMemberAccessor oldMember(oldObj, "gains");
			hkClassMemberAccessor newMember(newObj, "gains");

			hkVariant oldVariant;
			oldVariant.m_class = &(oldMember.getClassMember().getStructClass());
			oldVariant.m_object = oldMember.getAddress();

			hkVariant newVariant;
			newVariant.m_class = &(newMember.getClassMember().getStructClass());
			newVariant.m_object = newMember.getAddress();

			update_hkbFootIkGains( oldVariant, newVariant );
		}
	}

	static void update_hkbFootIkControlsModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// version the embedded control data
		{
			hkClassMemberAccessor oldMember(oldObj, "controlData");
			hkClassMemberAccessor newMember(newObj, "controlData");

			hkVariant oldVariant;
			oldVariant.m_class = &(oldMember.getClassMember().getStructClass());
			oldVariant.m_object = oldMember.getAddress();

			hkVariant newVariant;
			newVariant.m_class = &(newMember.getClassMember().getStructClass());
			newVariant.m_object = newMember.getAddress();

			update_hkbFootIkControlData( oldVariant, newVariant );
		}
	}

	static void update_hkbFootIkModifierLeg( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// convert max from cosine to degrees
		{
			hkClassMemberAccessor oldMember(oldObj, "cosineMaxKneeAngle");
			hkClassMemberAccessor newMember(newObj, "maxKneeAngleDegrees");

			const hkReal cosine = oldMember.asReal();
			const hkReal degrees = hkMath::acos( cosine ) * (1.0f / HK_REAL_DEG_TO_RAD);

			newMember.asReal() = degrees;
		}

		// convert min from cosine to degrees
		{
			hkClassMemberAccessor oldMember(oldObj, "cosineMinKneeAngle");
			hkClassMemberAccessor newMember(newObj, "minKneeAngleDegrees");

			const hkReal cosine = oldMember.asReal();
			const hkReal degrees = hkMath::acos( cosine ) * (1.0f / HK_REAL_DEG_TO_RAD);

			newMember.asReal() = degrees;
		}
	}

	static void update_hkbFootIkModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// version all of the legs
		{
			hkClassMemberAccessor oldMember(oldObj, "legs");
			hkClassMemberAccessor newMember(newObj, "legs");

			hkCompatUtil::versionArrayOfStructs( oldMember, newMember, update_hkbFootIkModifierLeg, tracker );
		}

		// version the gains
		{
			hkClassMemberAccessor oldMember(oldObj, "gains");
			hkClassMemberAccessor newMember(newObj, "gains");

			hkVariant oldVariant;
			oldVariant.m_class = &(oldMember.getClassMember().getStructClass());
			oldVariant.m_object = oldMember.getAddress();

			hkVariant newVariant;
			newVariant.m_class = &(newMember.getClassMember().getStructClass());
			newVariant.m_object = newMember.getAddress();

			update_hkbFootIkGains( oldVariant, newVariant );
		}
	}

	static void update_hkbJigglerGroup( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		hkClassMemberAccessor oldMember(oldObj, "alignMode");
		hkClassMemberAccessor newMember(newObj, "rotateBonesForSkinning");

		newMember.asBool() = (oldMember.asInt8() != 0);
	}

	static void update_hkbLookAtModifier( hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
	{
		// rename
		hkVersionUtil::renameMember( oldObj, "targetGain", newObj, "newTargetGain" );
		hkVersionUtil::renameMember( oldObj, "lookAtGain", newObj, "onOffGain" );

		// rename and convert to degrees
		{
			hkClassMemberAccessor oldMember(oldObj, "lookAtLimit");
			hkClassMemberAccessor newMember(newObj, "limitAngleDegrees");
			newMember.asReal() = oldMember.asReal() * ( 1.0f / HK_REAL_DEG_TO_RAD );
		}

		// rename and convert to degrees
		{
			hkClassMemberAccessor oldMember(oldObj, "lookUpAngle");
			hkClassMemberAccessor newMember(newObj, "lookUpAngleDegrees");
			newMember.asReal() = oldMember.asReal() * ( 1.0f / HK_REAL_DEG_TO_RAD );
		}
	}

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// common
	{ 0x8bdd3e9a, 0x8bdd3e9a, hkVersionRegistry::VERSION_VARIANT, "hkaBoneAttachment", HK_NULL },
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
	{ 0x1388d601, 0x1388d601, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributesAttribute", HK_NULL },
	{ 0xbff19005, 0xbff19005, hkVersionRegistry::VERSION_VARIANT, "hkCustomAttributes", HK_NULL },
	{ 0x57061454, 0x57061454, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

	// behavior
	{ 0x62eabf22, 0x5a2a86ba, hkVersionRegistry::VERSION_COPY, "hkbDemoConfig", HK_NULL },
	{ 0x307efb6d, 0xcd849481, hkVersionRegistry::VERSION_COPY, "hkbFootIkControlData", HK_NULL },
	{ 0x4a4cbd47, 0xbf1ce00e, hkVersionRegistry::VERSION_COPY, "hkbFootIkControlsModifier", update_hkbFootIkControlsModifier },
	BINARY_IDENTICAL( 0xff1f822c, 0x9af27949, "hkbFootIkGains" ),
	{ 0x8609b44f, 0xa3049f36, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifier", update_hkbFootIkModifier },
	{ 0x01ca3282, 0xbd3a7d99, hkVersionRegistry::VERSION_COPY, "hkbFootIkModifierLeg", HK_NULL },
	{ 0xc7e6bf96, 0x4eb19232, hkVersionRegistry::VERSION_COPY, "hkbGetUpModifier", HK_NULL },
	{ 0xfe5e54b6, 0xda4c7e80, hkVersionRegistry::VERSION_COPY, "hkbJigglerGroup", update_hkbJigglerGroup },
	{ 0x02587f55, 0x73a5a12a, hkVersionRegistry::VERSION_COPY, "hkbKeyframeBonesModifier", HK_NULL }, 
	{ 0x70729011, 0xb707333a, hkVersionRegistry::VERSION_COPY, "hkbLookAtModifier", update_hkbLookAtModifier }, 
	{ 0x1b87063b, 0x31ca69c9, hkVersionRegistry::VERSION_COPY, "hkbRotateCharacterModifier", HK_NULL },
	REMOVED("hkbRotateTranslateModifier"),

	// these are not versioned because they are not exported by HBT
	{ 0x07ccfea7, 0x38b6d406, hkVersionRegistry::VERSION_COPY, "hkbCatchFallModifier", HK_NULL },
	{ 0x6ec46ff5, 0x64f7d5a4, hkVersionRegistry::VERSION_COPY, "hkbCatchFallModifierHand", HK_NULL },
	{ 0xc92963cf, 0x80811815, hkVersionRegistry::VERSION_COPY, "hkbHandIkModifier", HK_NULL }, 
	{ 0x5e7f276b, 0xe9feed05, hkVersionRegistry::VERSION_COPY, "hkbHandIkModifierHand", HK_NULL }, 
	{ 0xc6acc99a, 0xc5b3a056, hkVersionRegistry::VERSION_COPY, "hkbReachModifier", HK_NULL },

	{ 0, 0, 0, HK_NULL, HK_NULL }
};

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok550b2Classes
#define HK_COMPAT_VERSION_TO hkHavok550r1Classes
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO

} // namespace hkCompat_hk550b2_hk550r1

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

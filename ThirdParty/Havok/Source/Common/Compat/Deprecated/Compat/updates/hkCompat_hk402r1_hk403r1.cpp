/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Compat/Deprecated/Compat/hkCompatUtil.h>
#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>
#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>

// We had to work around a shortcoming in the serialization framework wherein
// versioning functions are not applied to instances in arrays.  The classes
// hkbStateMachine and hkbStateMachine::StateInfo both include arrays of 
// TransitionInfos that need to be versioned.  So a versioning function has
// been written for arrays of TransitionInfos, and that function is called
// by the versioning functions for the two classes.

static void TransitionArray_402r1_403r1(
	hkClassMemberAccessor& oldArrayAccessor,
	hkClassMemberAccessor& newArrayAccessor )
{
	int oldSize = oldArrayAccessor.getClassMember().getArrayMemberSize();
	int newSize = newArrayAccessor.getClassMember().getArrayMemberSize();

	hkArray<char>* oldArray = reinterpret_cast< hkArray<char>* >( oldArrayAccessor.getAddress() );
	hkArray<char>* newArray = reinterpret_cast< hkArray<char>* >( newArrayAccessor.getAddress() );

	HK_ASSERT( 0x459ab345, oldArray->getSize() == newArray->getSize() );

	char* oldTransition = oldArray->begin();
	char* newTransition = newArray->begin();

	// We take a shortcut here.  In the old TransitionInfo, the hkbEvent is the first member
	// in the class, and its first member is the event ID.  In the new TransitionInfo, the
	// first member is the event ID.  Therefore we simply copy the first int between each old
	// and new TransitionInfo.  Nothing else needs to be done because the members have already been
	// copied by name.

	int count = oldArray->getSize();

	for( int i = 0; i < count; i++ )
	{
		*(reinterpret_cast<int*>( newTransition )) = *(reinterpret_cast<int*>( oldTransition ));

		oldTransition += oldSize;
		newTransition += newSize;
	}
}

static void hkbStateMachineStateInfo_402r1_403r1(
	hkVariant& oldObj,
	hkVariant& newObj,
	hkObjectUpdateTracker& )
{
	hkClassMemberAccessor oldTransitions(oldObj, "transitions");
	hkClassMemberAccessor newTransitions(newObj, "transitions");

	TransitionArray_402r1_403r1( oldTransitions, newTransitions );
}

static void hkbStateMachine_402r1_403r1(
	hkVariant& oldObj,
	hkVariant& newObj,
	hkObjectUpdateTracker& )
{
	hkClassMemberAccessor oldGlobalTransitions(oldObj, "globalTransitions");
	hkClassMemberAccessor newGlobalTransitions(newObj, "globalTransitions");

	TransitionArray_402r1_403r1( oldGlobalTransitions, newGlobalTransitions );

	hkClassMemberAccessor newReturnToPreviousStateEventId(newObj, "returnToPreviousStateEventId" );
	hkClassMemberAccessor newRandomTransitionEventId(newObj, "randomTransitionEventId" );

	newReturnToPreviousStateEventId.asInt32() = -1;
	newRandomTransitionEventId.asInt32() = -1;
}

namespace hkCompat_hk402r1_hk403r1
{

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }

	static const hkVersionRegistry::ClassAction s_updateActions[] =
	{
		// common
		{ 0x8bdd3e9a, 0x8bdd3e9a, hkVersionRegistry::VERSION_VARIANT, "hkBoneAttachment", HK_NULL },
		{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
		{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL }, 
		{ 0x3d43489c, 0x3d43489c, hkVersionRegistry::VERSION_VARIANT, "hkxMaterial", HK_NULL },
		{ 0x914da6c1, 0x914da6c1, hkVersionRegistry::VERSION_VARIANT, "hkxAttribute", HK_NULL },
		{ 0x1667c01c, 0x1667c01c, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeGroup", HK_NULL }, 
		{ 0x0a62c79f, 0x0a62c79f, hkVersionRegistry::VERSION_VARIANT, "hkxNode", HK_NULL }, 
		{ 0xe085ba9f, 0xe085ba9f, hkVersionRegistry::VERSION_VARIANT, "hkxMaterialTextureStage", HK_NULL },
		{ 0x57061454, 0x57061454, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },

		// hkbehavior
		{ 0x65d36ce9, 0x32c12475, hkVersionRegistry::VERSION_COPY, "hkbBlendingTransitionEffect", HK_NULL },
		{ 0xb9f995b7, 0xf61d9174, hkVersionRegistry::VERSION_COPY, "hkbClipGenerator", HK_NULL },
		{ 0x04055ac5, 0x99c6e0ba, hkVersionRegistry::VERSION_COPY, "hkbStateMachine", hkbStateMachine_402r1_403r1 },
		{ 0x7c043ced, 0x2d9f7a72, hkVersionRegistry::VERSION_COPY, "hkbStateMachineStateInfo", hkbStateMachineStateInfo_402r1_403r1 },
		{ 0xc39b1082, 0xb1043911, hkVersionRegistry::VERSION_COPY, "hkbStateMachineTransitionInfo", HK_NULL },
		{ 0x14b49cca, 0x8590cc86, hkVersionRegistry::VERSION_COPY, "hkbTransitionEffect", HK_NULL },
		{ 0x00432048, 0x0d52e940, hkVersionRegistry::VERSION_COPY, "hkbBlenderGenerator", HK_NULL },

		REMOVED( "hkbTransitionContext" ),

		{ 0, 0, 0, HK_NULL, HK_NULL }
	};

	static const hkVersionRegistry::ClassRename s_renames[] =
	{
		{ HK_NULL, HK_NULL }
	};

#define HK_COMPAT_VERSION_FROM hkHavok402r1Classes
#define HK_COMPAT_VERSION_TO hkHavok403r1Classes
#define HK_COMPAT_OPTIONAL_UPDATE_FUNC update

	extern hkVersionRegistry::UpdateDescription hkVersionUpdateDescription;

	static hkResult HK_CALL update(
		hkArray<hkVariant>& objectsInOut,
		hkObjectUpdateTracker& tracker )
	{
		hkCompatUtil::updateNamedVariantClassName( objectsInOut, s_renames, tracker );
		return hkVersionUtil::updateSingleVersion( objectsInOut, tracker, hkVersionUpdateDescription, hkVersionRegistry::getInstance().getClassNameRegistry(HK_COMPAT_VERSION_TO::VersionString) );
	}

#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO
} // namespace hkCompat_hk402r1_hk403r1

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

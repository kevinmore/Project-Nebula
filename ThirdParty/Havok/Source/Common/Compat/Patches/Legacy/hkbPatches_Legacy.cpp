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
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Container/String/hkStringBuf.h>

// Registration function is at the end of the file

static hkDataObject newEventProperty(const hkDataWorld* world)
{
	hkDataClass eventPropertyClass = world->findClass("hkbEventProperty");
	HK_ASSERT(0x7ea16f3d, !eventPropertyClass.isNull());
	return world->newObject(eventPropertyClass);
}

// versions a bone index array that has changed from hkArray<hkInt16> to hkbBoneIndexArray*
static void versionBoneIndexArray(hkDataObject& obj, const char* oldMemberName, const char* newMemberName )
{
	hkDataArray oldArray = obj[oldMemberName].asArray();
	const int sz = oldArray.getSize();

	if ( sz > 0 )
	{
		// create a new object
		hkDataObject newBoneIndexArrayObj = obj.getClass().getWorld()->newObject( obj.getClass().getWorld()->findClass("hkbBoneIndexArray") );
		obj[newMemberName] = newBoneIndexArrayObj;

		hkDataArray newArray = newBoneIndexArrayObj["boneIndices"].asArray();
		newArray.setSize(sz);

		for( int i = 0; i < sz; i++ )
		{
			newArray[i] = oldArray[i].asInt();
		}
	}
}

// versions a bone weight array that has changed from hkArray<hkReal> to hkbBoneWeightArray*
static void versionBoneWeightArray(hkDataObject& obj, const char* oldMemberName, const char* newMemberName)
{
	hkDataArray oldArray = obj[oldMemberName].asArray();
	const int sz = oldArray.getSize();

	if ( sz > 0 )
	{
		// create a new object
		hkDataObject newBoneWeightArrayObj = obj.getClass().getWorld()->newObject( obj.getClass().getWorld()->findClass("hkbBoneWeightArray") );
		obj[newMemberName].asObject() = newBoneWeightArrayObj;
		hkDataArray newArray = newBoneWeightArrayObj["boneWeights"].asArray();
		newArray.setSize(sz);

		for( int i = 0; i < sz; i++ )
		{
			newArray[i] = oldArray[i].asReal();
		}
	}
}


// versions an array of objects that has been moved inside another member
static void versionObjectArrayToEmbedded(hkDataObject& obj, const char* oldMemberName, const char* newMemberName, const char* embeddedArrayName, hkDataClass newMemberClass)
{
	hkDataArray oldArray = obj[oldMemberName].asArray();
	const int sz = oldArray.getSize();

	if ( sz > 0 )
	{
		// create a new object
		hkDataObject newArrayObj = obj.getClass().getWorld()->newObject( newMemberClass );
		obj[newMemberName] = newArrayObj;
		hkDataArray newArray = newArrayObj[embeddedArrayName].asArray();
		newArray.setSize(sz);

		for( int i = 0; i < sz; i++ )
		{
			newArray[i] = oldArray[i].asObject();
		}
	}
}

// versions a member that has changed from an hkbEvent to an hkbEventProperty
static void versionEventToEventProperty(hkDataObject& obj, const char* oldMemberName, const char* newMemberName)
{
	obj[newMemberName] = newEventProperty(obj.getClass().getWorld());
	obj[newMemberName].asObject()["id"] = obj[oldMemberName].asObject()["id"].asInt();
}

// versions a member that has changed from an integer event ID to an hkbEventProperty
static inline void versionEventIdToEventProperty(hkDataObject& obj, const char* oldMemberName, const char* newMemberName)
{
	obj[newMemberName] = newEventProperty(obj.getClass().getWorld());
	obj[newMemberName].asObject()["id"] = obj[oldMemberName].asInt();
}

static void hkbMoveBoneTowardTargetModifier_0_to_1(hkDataObject& obj)
{
	versionEventIdToEventProperty(obj, "old_eventToSendWhenTargetReached", "eventToSendWhenTargetReached");
}

static void hkbSplinePathGenerator_0_to_1(hkDataObject& obj)
{
	versionEventIdToEventProperty(obj, "pathEndEventId", "pathEndEvent");
}

static void hkbAttachmentModifier_0_to_1(hkDataObject& obj)
{
	versionEventIdToEventProperty(obj, "old_sendToAttacherOnAttach", "sendToAttacherOnAttach");
	versionEventIdToEventProperty(obj, "old_sendToAttacheeOnAttach", "sendToAttacheeOnAttach");
	versionEventIdToEventProperty(obj, "old_sendToAttacherOnDetach", "sendToAttacherOnDetach");
	versionEventIdToEventProperty(obj, "old_sendToAttacheeOnDetach", "sendToAttacheeOnDetach");
}

static void hkbTimerModifier_0_to_1(hkDataObject& obj)
{
	versionEventIdToEventProperty(obj, "eventIdToSend", "alarmEvent");
}

static void hkbClipGenerator_0_to_1(hkDataObject& obj)
{
	versionObjectArrayToEmbedded(obj, "old_triggers", "triggers", "triggers", obj.getClass().getWorld()->findClass("hkbClipTriggerArray") );
}

static void hkbTargetRigidBodyModifier_0_to_1(hkDataObject& obj)
{
	versionEventIdToEventProperty(obj, "old_eventToSend", "eventToSend");
	versionEventIdToEventProperty(obj, "old_eventToSendToTarget", "eventToSendToTarget");
	versionEventIdToEventProperty(obj, "closeToTargetEventId", "closeToTargetEvent");
}

static void hkbDetectCloseToGroundModifier_0_to_1(hkDataObject& obj)
{
	versionEventIdToEventProperty(obj, "closeToGroundEventId", "closeToGroundEvent");
}


static void hkbPositionRelativeSelectorGenerator_0_to_1(hkDataObject& obj)
{
	versionEventIdToEventProperty(obj, "fixPositionEventId", "fixPositionEvent");
}

static void hkbCheckRagdollSpeedModifier_0_to_1(hkDataObject& obj)
{
	versionEventIdToEventProperty(obj, "old_eventToSend", "eventToSend");
}

static void hkbCatchFallModifier_0_to_1(hkDataObject& obj)
{
	versionEventIdToEventProperty(obj, "catchFallDoneEventId", "catchFallDoneEvent");
	versionBoneIndexArray(obj, "old_spineIndices", "spineIndices");
}

static void hkbRigidBodyRagdollModifier_0_to_1(hkDataObject& obj)
{
	versionBoneIndexArray(obj, "old_keyframedBonesList", "keyframedBonesList");
}

static void hkbPoweredRagdollControlsModifier_0_to_1(hkDataObject& obj)
{
	versionBoneWeightArray(obj, "old_boneWeights", "boneWeights");
}

static void hkbKeyframeBonesModifier_0_to_1(hkDataObject& obj)
{
	versionBoneIndexArray(obj, "old_keyframedBonesList", "keyframedBonesList");
}

static void hkbBlenderGeneratorChild_0_to_1(hkDataObject& obj)
{
	versionBoneWeightArray(obj, "old_boneWeights", "boneWeights");
}

static void hkbJigglerGroup_0_to_1(hkDataObject& obj)
{
	versionBoneIndexArray(obj, "old_boneIndices", "boneIndices");
}

static void hkbStateMachine_0_to_1(hkDataObject& obj)
{
	versionObjectArrayToEmbedded(obj, "globalTransitions", "wildcardTransitions", "transitions", obj.getClass().getWorld()->findClass("hkbStateMachineTransitionInfoArray") );
}

static void hkbStateMachineStateInfo_0_to_1(hkDataObject& obj)
{
	versionEventToEventProperty(obj, "old_enterNotifyEvent", "enterNotifyEvent");
	versionEventToEventProperty(obj, "old_exitNotifyEvent", "exitNotifyEvent");
}

static void hkbFootIkModifierLeg_0_to_1(hkDataObject& obj)
{
	versionEventIdToEventProperty(obj, "ungroundedEventId", "ungroundedEvent");
}

static void hkbClipTrigger_0_to_1(hkDataObject& obj)
{
	versionEventToEventProperty(obj, "old_event", "event");
}

static void hkbBehaviorGraphData_0_to_1(hkDataObject& obj)
{
	// In version 0, the word-sized initial values of the variables were stored in the hkbVariableInfo and the
	// quad initial values were stored in the member m_quadVariableInitialValues.  In version 1 we need to move
	// the initial values into m_variableInitialValues.  Note that the word-sized initial values index into 
	// the quad initial values for quad-sized variables.  In version 0, this included Object Pointer variables.
	// In version 1, the indices of Object Pointer (Variant) variables are into a new list for variables.
	// So those indices need to be recomputed.

	hkDataArray variableInfos = obj["variableInfos"].asArray();
	const int numVars = variableInfos.getSize();

	int numQuads = 0;
	int numVariants = 0;

	hkDataArray oldQuadInitialValues = obj["quadVariableInitialValues"].asArray();

	// create a new object for initialValues (a new member)
	// create a new object
	hkDataClass variableValueSetClass = obj.getClass().getWorld()->findClass("hkbVariableValueSet");
	hkDataObject initialValues = obj.getClass().getWorld()->newObject( variableValueSetClass );
	obj["variableInitialValues"] = initialValues;
	hkDataArray quadInitialValues = initialValues["quadVariableValues"].asArray();
	hkDataArray wordInitialValues = initialValues["wordVariableValues"].asArray();

	wordInitialValues.setSize(numVars);

	enum
	{
		VARIABLE_TYPE_BOOL,
		VARIABLE_TYPE_INT8,
		VARIABLE_TYPE_INT16,
		VARIABLE_TYPE_INT32,
		VARIABLE_TYPE_REAL,
		VARIABLE_TYPE_POINTER,
		VARIABLE_TYPE_VECTOR3,
		VARIABLE_TYPE_VECTOR4,
		VARIABLE_TYPE_QUATERNION,
	};

	for( int i = 0; i < numVars; i++ )
	{
		switch( variableInfos[i].asObject()["type"].asInt() )
		{
		case VARIABLE_TYPE_INT8:
		case VARIABLE_TYPE_INT16:
		case VARIABLE_TYPE_INT32:
		case VARIABLE_TYPE_REAL:
			wordInitialValues[i].asObject()["value"] = variableInfos[i].asObject()["initialValue"].asObject()["value"].asInt();
			break;
		case VARIABLE_TYPE_POINTER:
			numVariants++;
			break;
		case VARIABLE_TYPE_VECTOR3:
		case VARIABLE_TYPE_VECTOR4:
		case VARIABLE_TYPE_QUATERNION:
			wordInitialValues[i].asObject()["value"] = numQuads;
			quadInitialValues.setSize(numQuads+1);
			// the old quad initial values included object pointers (variants) so we need to factor them
			// in when indexing into the array
			quadInitialValues[numQuads].setVec( oldQuadInitialValues[numQuads+numVariants].asVec(4), 4 );
			numQuads++;
			break;
		default:
			HK_ASSERT2( 0x59a7d761, false, "unexpected case" );
		}
	}

	hkDataArray variantInitialValues = initialValues["variantVariableValues"].asArray();
	variantInitialValues.setSize(numVariants);
}

static bool isBindable(hkDataClass c)
{
	while( !c.isNull() )
	{
		if ( hkString::strCmp( c.getName(), "hkbBindable" ) )
		{
			return true;
		}

		c = c.getParent();
	}

	return false;
}

static void versionBindingPath(hkDataObject obj, const char* path, hkDataObject& bindable, const char*& pathFromBindable)
{
	pathFromBindable = HK_NULL;

	while( ( path != HK_NULL ) && ( *path != '\0' ) )
	{
		if ( isBindable( obj.getClass() ) )
		{
			bindable = obj;
			pathFromBindable = path;
		}

		int colonIndex = -1;
		int i = 0;

		while( path[i] != '\0' && path[i] != '/' )
		{
			if ( path[i] == ':' )
			{
				colonIndex = i;
			}

			i++;
		}

		int memberLen = ( colonIndex != -1 ) ? colonIndex : i;

		HK_ASSERT2( 0xa781b734, memberLen > 0, "binding path ended prematurely" );

		int arrayIndex = -1;

		if ( colonIndex != -1 )
		{
			const char* indexString = path + colonIndex + 1;
			arrayIndex = hkString::atoi( indexString );
		}

		hkLocalBuffer<char> memberName( memberLen + 1 );
		hkString::strNcpy( memberName.begin(), path, memberLen );
		memberName[memberLen] = '\0';

		if ( obj.getClass().getMemberIndexByName( memberName.begin() ) == -1 )
		{
			HK_ASSERT2( 0xd891ab6e, false, "member not found when versioning binding path" );
			break;
		}

		hkDataObject::Type type = obj[memberName.begin()].getType();
		hkDataObject::Type basicType = type->getParent();

		if ( ( basicType->isPointer() ) || ( basicType->isClass()) )
		{
			bool isArray = type->isArray() || type->isTuple();

			if ( isArray )
			{
				HK_ASSERT2( 0x6723b761, arrayIndex != -1, "array index mismatch in binding path" );

				obj = obj[memberName.begin()].asArray()[arrayIndex].asObject();
			}
			else
			{
				obj = obj[memberName.begin()].asObject();
			}
		}
		else
		{
			break;
		}

		path = path + i;

		// skip the slash
		if ( *path == '/' )
		{
			path++;
		}
	}

	HK_ASSERT2( 0xc891bd67, pathFromBindable != HK_NULL, "binding path versioning failed" );
}

static void hkbNode_0_to_1(hkDataObject& obj)
{
	obj["variableBindingSet"] = obj["old_variableBindingSet"].asObject();

	// figure out which bindings are potentially to be left with this node
	{
		hkDataObject variableBindingSet = obj["variableBindingSet"].asObject();

		if ( !variableBindingSet.isNull() )
		{
			hkDataArray bindingsArray = variableBindingSet["bindings"].asArray();
			int sz = bindingsArray.getSize();

			for( int i = 0; i < sz; i++ )
			{
				hkDataObject binding = bindingsArray[i].asObject();
				hkDataObject boundObj = binding["object"].asObject();

				// If the object to bind to is the node itself, then this binding will not need to be moved.
				// We indicate this case by negating the variable index after adding one (to avoid negating 0).
				if ( boundObj == obj )
				{
					binding["variableIndex"] = -(1 + binding["variableIndex"].asInt());
				}
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// When versioning bindings, some of the bindings need to be moved from one binding set to another.
// This creates a problem because the bindings don't know which set they belong to when versioning.
// We solve this problem in three stages:
// 1) We first version hkbNodes.  A node tags all bindings whose bound object is the node itself.
//    These bindings may not need to be moved (they will still need to be moved if their paths
//    go beyond another hkbBindable.  The tagging is done by negating the variable index after adding
//    1 to it (to take care of 0).
// 2) Next we version the bindings.  Each binding is marked in it's bindingType member as either
//    BINDING_TYPE_KEEP or BINDING_TYPE_REMOVE depending on whether the binding needed to be added
//    to a different bindable other than the node itself.
// 3) Finally, we version the binding sets, where we remove all bindings marked as BINDING_TYPE_REMOVE.
//////////////////////////////////////////////////////////////////////////////////////////////////////////

const int BINDING_TYPE_KEEP = 0; // should be set to hkbVariableBindingSet::BINDING_TYPE_VARIABLE
const int BINDING_TYPE_REMOVE = -1;

static void hkbVariableBindingSetBinding_0_to_1(hkDataObject& obj)
{
	bool boundToOwnNode = false;

	// See if this one is tagged as being bound to the node itself, which is indicated by
	// a negative variable index.
	if ( obj["variableIndex"].asInt() < 0 )
	{
		boundToOwnNode = true;

		// put the index back to what it is supposed to be
		obj["variableIndex"] = -obj["variableIndex"].asInt() - 1;
	}

	// convert the binding to one that is from the nearest hkbBindable
	const char* originalMemberPath = obj["memberPath"].asString();
	hkStringBuf memberPath = originalMemberPath;
	hkDataObject boundObj = obj["object"].asObject();
	const char* pathFromBindable = HK_NULL;
	hkDataObject bindable( HK_NULL );
	{
		// Take care of some special cases where the paths have changed due to the introduction of
		// hkbBoneIndexArray and hkbBoneWeightArray.
		{
			const char* className = obj.getClass().getName();

			if (	( 0 == hkString::strCmp( className, "hkbPoweredRagdollControlsModifier" ) ) ||
				( 0 == hkString::strCmp( className, "hkbBlenderGenerator" ) ) )
			{
				// There are several cases here.  Although the binding will have been placed on the node,
				// it may be rooted at the node or at the hkbBlenderGeneratorChild.  We use replaceInplace to
				// handle both cases.
				memberPath.replace( "boneWeights", "boneWeights/boneWeights", hkStringBuf::REPLACE_ONE );
			}
			else if (	( 0 == hkString::strCmp( className, "hkbKeyframeBonesModifier" ) ) ||
				( 0 == hkString::strCmp( className, "hkbRigidBodyRagdollModifier" ) ) )
			{
				memberPath.replace( "keyframedBonesList", "keyframedBonesList/boneIndices", hkStringBuf::REPLACE_ONE );
			}
			else if ( 0 == hkString::strCmp( className, "hkbJigglerModifier" ) )
			{
				// As above, there are two cases that we need to consider, so we use replaceInplace.
				memberPath.replace( "boneIndices", "boneIndices/boneIndices", hkStringBuf::REPLACE_ONE );
			}

			versionBindingPath( boundObj, memberPath.cString(), bindable, pathFromBindable );
		}

		HK_ASSERT2( 0x71832b8f, !bindable.isNull() && isBindable(bindable.getClass()), "binding is to a non-bindable object" );
	}

	// see if we are keeping the old binding or adding a new one
	if ( boundToOwnNode && ( boundObj == bindable ) )
	{
		// the string should be the original string in this case
		HK_ASSERT( 0x571bc78d, 0 == hkString::strCmp( obj["memberPath"].asString(), pathFromBindable ) );

		// the binding stays where it was
		obj["bindingType"] = BINDING_TYPE_KEEP;
	}
	else
	{
		// the binding is being moved
		obj["bindingType"] = BINDING_TYPE_REMOVE;

		// add the binding to the appropriate hkbBindable

		hkDataObject variableBindingSetObj = bindable["variableBindingSet"].asObject();

		// create an hkbVariableBindingSet if there isn't one
		if ( variableBindingSetObj.isNull() )
		{
			variableBindingSetObj = obj.getClass().getWorld()->newObject( bindable.getClass().getWorld()->findClass( "hkbVariableBindingSet" ) );
			bindable["variableBindingSet"] = variableBindingSetObj;
		}

		hkDataArray bindingsArray = variableBindingSetObj["bindings"].asArray();

		// add the new binding to the array
		{
			hkDataObject newBinding = obj.getClass().getWorld()->newObject( bindingsArray.getClass() );
			newBinding["bitIndex"] = obj["bitIndex"].asInt();
			newBinding["variableIndex"] = obj["variableIndex"].asInt();
			newBinding["memberPath"] = pathFromBindable;
			// the new binding is a keeper
			newBinding["bindingType"] = BINDING_TYPE_KEEP;

			const int sz = bindingsArray.getSize();
			bindingsArray.setSize( sz + 1 );
			bindingsArray[sz] = newBinding;
		}
	}
}

static void hkbVariableBindingSet_0_to_1(hkDataObject& obj)
{
	// remove all of the bindings that are not marked KEEP
	{
		hkDataArray bindingsArray = obj["bindings"].asArray();

		const int sz = bindingsArray.getSize();
		int toIndex = 0;
		for( int i = 0; i < sz; i++ )
		{
			hkDataObject obji = bindingsArray[i].asObject();

			// make sure we set the variableIndex back to something reasonable
			HK_ASSERT( 0x6512b4f8, obji["variableIndex"].asInt() >= 0 );

			if( obji["bindingType"].asInt() == BINDING_TYPE_KEEP )
			{
				if ( i != toIndex )
				{
					bindingsArray[toIndex] = bindingsArray[i].asObject();
				}

				toIndex++;
			}
		}

		bindingsArray.setSize( toIndex );
	}
}

//=======
// 650r1
//=======

static void hkbStateMachineStateInfo_1_to_2(hkDataObject& obj)
{
	versionObjectArrayToEmbedded(obj, "old_transitions", "transitions", "transitions", obj.getClass().getWorld()->findClass("hkbStateMachineTransitionInfoArray") );
}


static void hkbDemoConfig_0_to_1(hkDataObject& obj)
{
	hkDataArray stickVariables = obj["stickVariables"].asArray();
	hkDataArray old_stickVariables = obj["old_stickVariables"].asArray();

	for( int i = 0; i < 4; i++ )
	{
		stickVariables[i] = old_stickVariables[i].asObject();
	}
}

static void hkbVariableBindingSet_1_to_2(hkDataObject& obj)
{
	hkDataArray bindingsArray = obj["bindings"].asArray();

	const int sz = bindingsArray.getSize();

	for( int i = 0; i < sz; i++ )
	{
		hkDataObject obji = bindingsArray[i].asObject();
		const char* memberPath = obji["memberPath"].asString();

		if ( 0 == hkString::strCmp( memberPath, "enable" ) )
		{
			obj["indexOfBindingToEnable"] = i;
			return;
		}
	}

	obj["indexOfBindingToEnable"] = -1;
}

//==========
// post-6.5
//==========


static void hkbClipGenerator_1_to_2(hkDataObject& obj)
{
	// We removed MODE_USER_CONTROLLED_LOOPING(3).
	// We need to decrement all modes higher than 3.
	// we also want to replace MODE_USER_CONTROLLED_LOOPING(3)
	// with MODE_USER_CONTROLLED(2).  This code does both.

	int mode = obj["mode"].asInt();

	if ( mode >= 3 )
	{
		mode--;
	}

	obj["mode"] = mode;
}

//==============
// 7.0 release 
//==============

static void hkbRagdollController_0_to_1(hkDataObject& obj)
{
	obj["worldFromModelModeSetup"].asObject()["poseMatchingBone0"] = obj["poseMatchingBone0"];
	obj["worldFromModelModeSetup"].asObject()["poseMatchingBone1"] = obj["poseMatchingBone1"];
	obj["worldFromModelModeSetup"].asObject()["poseMatchingBone2"] = obj["poseMatchingBone2"];
	obj["worldFromModelModeSetup"].asObject()["mode"] = obj["worldFromModelMode"];
}

static void hkbRagdollModifier_1_to_2(hkDataObject& obj)
{
	obj["worldFromModelModeSetup"].asObject()["poseMatchingBone0"] = obj["poseMatchingBone0"];
	obj["worldFromModelModeSetup"].asObject()["poseMatchingBone1"] = obj["poseMatchingBone1"];
	obj["worldFromModelModeSetup"].asObject()["poseMatchingBone2"] = obj["poseMatchingBone2"];
	obj["worldFromModelModeSetup"].asObject()["mode"] = obj["worldFromModelMode"];
}

static void hkbPoweredRagdollControlsModifier_3_to_4(hkDataObject& obj)
{
	obj["worldFromModelModeSetup"].asObject()["poseMatchingBone0"] = obj["poseMatchingBone0"];
	obj["worldFromModelModeSetup"].asObject()["poseMatchingBone1"] = obj["poseMatchingBone1"];
	obj["worldFromModelModeSetup"].asObject()["poseMatchingBone2"] = obj["poseMatchingBone2"];
	obj["worldFromModelModeSetup"].asObject()["mode"] = obj["worldFromModelMode"];
}

static void hkbDemoConfig_1_to_2(hkDataObject& obj)
{
	obj["stickVariables"].asArray().reserve(12);
	obj["stickVariables"].asArray().setSize(12);

	for( int i = 0; i < 12; i++ )
	{
		obj["stickVariables"].asArray()[i] = obj["old_stickVariables"].asArray()[i].asObject();
	}
}

static void hkbGetUpModifier_1_to_2(hkDataObject& obj)
{
	obj["groundNormal"] = obj["surfaceNormal"].asVector4();
	obj["alignWithGroundDuration"] = obj["correctToGroundTime"].asReal();
}

void HK_CALL registerBehaviorPatches_Legacy(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/Legacy/hkbPatches_Legacy.cxx>
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

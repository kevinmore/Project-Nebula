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
#include <Common/Base/Container/String/hkStringBuf.h>

// Registration function is at the end of the file

// versions a single object that has been moved inside another member which holds as array of these objects.
static void versionObjectToEmbedded(hkDataObject& obj, const char* oldMemberName, const char* newMemberName, const char* embeddedArrayName, hkDataClass newMemberClass)
{
	// create a new object
	hkDataObject newArrayObj = obj.getClass().getWorld()->newObject( newMemberClass );
	obj[newMemberName] = newArrayObj;

	hkDataArray newArray = newArrayObj[embeddedArrayName].asArray();
	newArray.setSize(1);
	newArray[0] = obj[oldMemberName].asObject();
}

// This function is used when you've renamed a member and you want to make sure the bindings on the node are renamed accordingly.
static void replacePropertyNameInBindings( hkDataObject& variableBindingSet, const char* oldName, const char* newName )
{
	if ( !variableBindingSet.isNull() )
	{
		hkDataArray bindingsArray = variableBindingSet["bindings"].asArray();
		int sz = bindingsArray.getSize();

		for( int i = 0; i < sz; i++ )
		{
			hkDataObject binding = bindingsArray[i].asObject();
			const char* oldMemberPath = binding["memberPath"].asString();

			if ( hkString::beginsWith( oldMemberPath, oldName ) )
			{
				hkStringBuf s;
				s.printf("%s%s", newName, oldMemberPath + hkString::strLen(oldName));
				binding["memberPath"] = s.cString();
			}
		}
	}
}

// This function replicates a binding to one member for a new member.  This is used if you introduce a new member
// but you want it bound to the same variable as an existing member.
static void replicateBindings( hkDataObject& variableBindingSet, const char* oldName, const char* newName )
{
	if ( !variableBindingSet.isNull() )
	{
		hkDataArray bindingsArray = variableBindingSet["bindings"].asArray();
		int sz = bindingsArray.getSize();

		for( int i = 0; i < sz; i++ )
		{
			hkDataObject binding = bindingsArray[i].asObject();
			const char* oldMemberPath = binding["memberPath"].asString();

			if ( hkString::beginsWith( oldMemberPath, oldName ) )
			{
				hkStringBuf s;
				s.printf("%s%s", newName, oldMemberPath + hkString::strLen(oldName));
				int j = bindingsArray.getSize();
				bindingsArray.setSize( j + 1 );
				bindingsArray[j] = binding;
				bindingsArray[j].asObject()["memberPath"] = s.cString();
			}
		}
	}
}

static void hkbLookAtModifier_2_to_3(hkDataObject& obj)
{
	// by default we assume that the neck and the head are aligned
	obj["neckForwardLS"] = obj["headForwardLS"].asVector4();

	hkDataObject variableBindingSet = obj["variableBindingSet"].asObject();

	replacePropertyNameInBindings( variableBindingSet, "headForwardHS", "headForwardLS" );
	replacePropertyNameInBindings( variableBindingSet, "headRightHS", "neckRightLS" );
	replicateBindings( variableBindingSet, "headForwardLS", "neckForwardLS" );
}

static void hkbStateMachineStateInfo_3_to_4(hkDataObject& obj)
{
	versionObjectToEmbedded( obj, "enterNotifyEvent", "enterNotifyEvents", "events", obj.getClass().getWorld()->findClass("hkbStateMachineEventPropertyArray") );
	versionObjectToEmbedded( obj, "exitNotifyEvent", "exitNotifyEvents", "events", obj.getClass().getWorld()->findClass("hkbStateMachineEventPropertyArray") );
}

void HK_CALL registerBehaviorPatches_2010_1(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2010_1/hkbPatches_2010_1.cxx>
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

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Base/Reflection/hkClassMemberAccessor.h>
#include <Common/Base/System/Io/Writer/Buffered/hkBufferedStreamWriter.h>
#include <Common/Serialize/Copier/hkObjectCopier.h>

extern const hkClass hkClassMemberClass;

const char* HK_CALL hkVersionUtil::getCurrentVersion()
{
	return HAVOK_SDK_VERSION_STRING;
}

void HK_CALL hkVersionUtil::renameMember( hkVariant& oldObj, const char* oldName, hkVariant& newObj, const char* newName )
{
	hkClassMemberAccessor oldMember(oldObj, oldName);
	hkClassMemberAccessor newMember(newObj, newName);
	if( oldMember.isOk() && newMember.isOk() )
	{
		HK_ASSERT(0x2912efc3, oldMember.getClassMember().getSizeInBytes() == newMember.getClassMember().getSizeInBytes() );
		hkString::memCpy( newMember.asRaw(), oldMember.asRaw(), newMember.getClassMember().getSizeInBytes() );
	}
}

void HK_CALL hkVersionUtil::copyDefaults( void* obj, const hkClass& oldClass, const hkClass& newClass )
{
	hkBufferedStreamWriter writer( obj, newClass.getObjectSize(), false );
	for( int memIndex = 0; memIndex < newClass.getNumMembers(); ++memIndex )
	{
		const hkClassMember& newMem = newClass.getMember(memIndex);
		if( oldClass.getMemberByName(newMem.getName()) == HK_NULL )
		{
			writer.seek( newMem.getOffset(), hkStreamWriter::STREAM_SET );
			newClass.getDefault( memIndex, &writer );
		}
	}
}

void HK_CALL hkVersionUtil::recomputeClassMemberOffsets( hkClass*const* classes, int classVersion )
{
	HK_ASSERT(0x67b28c08, classVersion > 0);
	hkClass::updateMetadataInplace( const_cast<hkClass**>(classes), classVersion );

	hkStructureLayout layout;
	hkPointerMap<const hkClass*, int> classesDone;
	for( int i = 0; classes[i] != HK_NULL; ++i )
	{
		layout.computeMemberOffsetsInplace( *classes[i], classesDone );
	}
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

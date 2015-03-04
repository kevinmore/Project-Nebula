/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/hkVisualDebugger.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/Process/hkInspectProcess.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeIStream.h>
#include <Common/Visualize/Serialize/hkObjectSerialize.h>

int hkInspectProcess::m_tag = 0;

const class hkClass* hkInspectProcess::s_specialClassClasses[] = {&hkClassClass, &hkClassMemberClass, &hkClassEnumClass, &hkClassEnumItemClass};
const int hkInspectProcess::s_numSpecialClassClasses = HK_COUNT_OF(s_specialClassClasses);

static void HK_CALL myTrackedObjectCallback( void* ptr, const hkClass* klass, hkBool wasAdded, hkUlong tagId, void* userCallbackHandle )
{
	hkInspectProcess* proc = (hkInspectProcess*)userCallbackHandle;
	if (wasAdded)
	{
		HK_ASSERT(0x210f3135, klass && ptr );
		proc->addTopLevelObject(ptr, *klass, tagId);
	}
	else
	{
		HK_ASSERT(0x77833b63, ptr );
		proc->removeTopLevelObject( ptr );
	}
}


hkProcess* HK_CALL hkInspectProcess::create(const hkArray<hkProcessContext*>& contexts)
{
	return new hkInspectProcess(contexts); // doesn't require a context 
}

void HK_CALL hkInspectProcess::registerProcess()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkInspectProcess::hkInspectProcess(const hkArray<hkProcessContext*>& contexts)
: hkProcess( true) /* user selectable */
{
	m_vdb = contexts.getSize() > 0 ?contexts[0]->m_owner : HK_NULL;
	if (m_vdb)
	{
		m_vdb->addReference();
		m_vdb->addTrackedObjectCallback( myTrackedObjectCallback, this );
	}
	m_cache = new hkPlatformObjectWriter::Cache;
}

hkInspectProcess::~hkInspectProcess()
{
	if (m_vdb)	
	{
		// for all tracked objs sendTopLevelObject
		const hkArray<hkVisualDebuggerTrackedObject>& trackedObjs = m_vdb->getTrackedObjects();
		for (int ti =0; ti < trackedObjs.getSize(); ++ti)
		{
			const hkVisualDebuggerTrackedObject& to = trackedObjs[ti];
			removeTopLevelObject( to.m_ptr );
		}

		m_vdb->removeTrackedObjectCallback( myTrackedObjectCallback );
		m_vdb->removeReference();
	}
	m_cache->removeReference();
}

extern const hkClass hkClassClass;
void hkInspectProcess::init()
{
	if (m_vdb) // handle the case of no contexts
	{
		// write what we see as a hkClass
		{
			int packetSize = 1 + 8 + 4 + (2 + 8 * s_numSpecialClassClasses);
			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_SETUP);
			
			// the class class. We need to use the cache if it is present as the server will
			// only ever see the cached version
			const hkClass* sendClassClass = &hkClassClass;
			if(m_cache)
			{
				// This is also hardcoded into hkObjectSerialize::writeObject
				hkStructureLayout pcLayout( hkStructureLayout::MsvcWin32LayoutRules );
				sendClassClass = m_cache->get(&hkClassClass, pcLayout);
			}
			m_outStream->write64u( (hkUlong) sendClassClass ); // classClassID, so that the clients know what to use to query for class data itself

			// the layout of this server  (4100 etc)
			m_outStream->write8u( hkStructureLayout::HostLayoutRules.m_bytesInPointer );
			m_outStream->write8u( hkStructureLayout::HostLayoutRules.m_littleEndian );
			m_outStream->write8u( hkStructureLayout::HostLayoutRules.m_reusePaddingOptimization );
			m_outStream->write8u( hkStructureLayout::HostLayoutRules.m_emptyBaseClassOptimization );

			m_outStream->write16u(s_numSpecialClassClasses);
			for(int i=0; i<s_numSpecialClassClasses;i++)
			{
				sendClassClass = s_specialClassClasses[i];
				if(m_cache)
				{
					hkStructureLayout pcLayout( hkStructureLayout::MsvcWin32LayoutRules );
					sendClassClass = m_cache->get(s_specialClassClasses[i], pcLayout);
				}
				m_outStream->write64u(hkUlong(sendClassClass));
			}

		}

		// for all tracked objs sendTopLevelObject
		const hkArray<hkVisualDebuggerTrackedObject>& trackedObjs = m_vdb->getTrackedObjects();
		for (int ti =0; ti < trackedObjs.getSize(); ++ti)
		{
			const hkVisualDebuggerTrackedObject& to = trackedObjs[ti];
			addTopLevelObject( to.m_ptr, *(to.m_class), to.m_tag);
		}
	}
}

void hkInspectProcess::step(hkReal frameTimeInMs)
{
	hkPointerMap<const void*, const hkClass*> writtenObjects;

	for (int au=0; au < m_autoUpdateList.getSize(); ++au)
	{
		writtenObjects.clear();
		writeObject( m_autoUpdateList[au].obj, *(m_autoUpdateList[au].klass), false, writtenObjects );
	}
}


struct _DummySimpleArray
{
	char* p;
	int s;
};

static void* _FindMemberItem(void* memberPtr, const hkArray<hkUint16>& memberPath, int curMemberIndex, hkClassMember::Type t, hkClassMember::Type subType, const hkClassMember* member)
{	
	switch ( t )
	{
		// XXX Todo: handle carrays here,
		// for the momen assumed indivisable:
	case hkClassMember::TYPE_BOOL:
	case hkClassMember::TYPE_CHAR: 
	case hkClassMember::TYPE_INT8:
	case hkClassMember::TYPE_UINT8:
	case hkClassMember::TYPE_INT16:
	case hkClassMember::TYPE_UINT16:
	case hkClassMember::TYPE_INT32:
	case hkClassMember::TYPE_UINT32:
	case hkClassMember::TYPE_INT64:
	case hkClassMember::TYPE_UINT64:
	case hkClassMember::TYPE_ULONG:
	case hkClassMember::TYPE_REAL:
	case hkClassMember::TYPE_HALF:
	case hkClassMember::TYPE_VECTOR4:
	case hkClassMember::TYPE_QUATERNION:
	case hkClassMember::TYPE_ENUM:
	case hkClassMember::TYPE_FLAGS:

		HK_ASSERT(0xaa0ba0b, memberPath.getSize() == (curMemberIndex + 1) );
		return memberPtr; // end of the line.

	case hkClassMember::TYPE_MATRIX3:
	case hkClassMember::TYPE_ROTATION:
	case hkClassMember::TYPE_QSTRANSFORM:
	case hkClassMember::TYPE_MATRIX4:
	case hkClassMember::TYPE_TRANSFORM:
		{
			int childIndex = memberPath[curMemberIndex + 1];
			char* childPtr = (char*)( ((float*)memberPtr) + (childIndex*4) );
			return _FindMemberItem( childPtr, memberPath, curMemberIndex+1, hkClassMember::TYPE_VECTOR4, hkClassMember::TYPE_VOID, member ); 
		}

	case hkClassMember::TYPE_POINTER:
		return HK_NULL; // not handled yet

	case hkClassMember::TYPE_ARRAY:
	case hkClassMember::TYPE_SIMPLEARRAY:
		{
			int childIndex = memberPath[curMemberIndex + 1];
			_DummySimpleArray* array= (_DummySimpleArray*)(memberPtr);
			char* childPtr = array->p + (member->getArrayMemberSize()*childIndex);
			return _FindMemberItem(childPtr, memberPath, curMemberIndex+1, member->getSubType(), hkClassMember::TYPE_VOID, member );
		}

	case hkClassMember::TYPE_STRUCT:
		{
			const hkClass* ptrClass = ( member && member->hasClass())? &(member->getStructClass()) : HK_NULL;
			if (ptrClass)
			{
				int childIndex = memberPath[curMemberIndex + 1];
				const hkClassMember& mem = ptrClass->getMember(childIndex);
				return _FindMemberItem( (((char*)memberPtr) + mem.getOffset()), memberPath, curMemberIndex+1, mem.getType(), mem.getSubType(), &mem );
			}
			break;
		}

	case hkClassMember::TYPE_VARIANT:
	case hkClassMember::TYPE_INPLACEARRAY:
	case hkClassMember::TYPE_HOMOGENEOUSARRAY:
	default: // includes hkZero
		{
			break;
		}
	}

	return HK_NULL;
}

static hkUint8 _inspectProcess_cmds[] = { 
	hkVisualDebuggerProtocol::HK_ADD_TOPLEVEL, 
	hkVisualDebuggerProtocol::HK_REMOVE_TOPLEVEL,
	hkVisualDebuggerProtocol::HK_REQUEST_OBJECT, 
	hkVisualDebuggerProtocol::HK_ADD_OBJECT,
	hkVisualDebuggerProtocol::HK_UPDATE_MEMBER
};

void hkInspectProcess::getConsumableCommands( hkUint8*& commands, int& numCommands )
{
	commands = _inspectProcess_cmds;
	numCommands	= sizeof(_inspectProcess_cmds);
}

void hkInspectProcess::consumeCommand( hkUint8 command  )
{
	switch (command)
	{
	case hkVisualDebuggerProtocol::HK_ADD_TOPLEVEL:
	case hkVisualDebuggerProtocol::HK_REMOVE_TOPLEVEL:
	case hkVisualDebuggerProtocol::HK_ADD_OBJECT:
		// really only impl on client.
		break;

		// client update of a specific server data item
	case hkVisualDebuggerProtocol::HK_UPDATE_MEMBER:
		{
			// IDs
			hkUint64 objectID = m_inStream->read64u();
			hkUint64 classID = m_inStream->read64u();
			
			// Member Path
			hkUint16 sizeOfMemberPath = m_inStream->read16u();
			hkArray<hkUint16> memberPath(sizeOfMemberPath);
			memberPath.setSize(sizeOfMemberPath);
			for (int mp=0; mp < sizeOfMemberPath; ++mp)
			{
				memberPath[mp] = m_inStream->read16u();
			}

			// Data packet, in this platform form.
			hkUint32 sizeOfData = m_inStream->read32u();
			hkArray<hkUint8> data(sizeOfData);
			data.setSize(sizeOfData);
			m_inStream->readRaw(data.begin(), sizeOfData);

			// Apply it if it all looks valid
			if ( (objectID > 0) && (classID > 0) && (memberPath.getSize() > 0) && (sizeOfData > 0) )
			{
				void* object = (void*)(hkUlong)( objectID );
				hkClass* klass = (hkClass*)(hkUlong)( classID );
				const hkClassMember& mem = klass->getMember(memberPath[0]);
				void* destinationPtr = _FindMemberItem( ((char*)object) + mem.getOffset(), memberPath, 0, mem.getType(), mem.getSubType(), &mem );
				if (destinationPtr)
				{
					hkString::memCpy( destinationPtr, data.begin(), sizeOfData); // .. and hope for the best ;)
				}
			}
		}

		break;

	default:
		break;

	case hkVisualDebuggerProtocol::HK_REQUEST_OBJECT:
		{
			hkUint64 objectID = m_inStream->read64u();
			hkUint64 classID = m_inStream->read64u();
			hkUint8 updateFlags = m_inStream->read8u();

			if ( (objectID > 0) && (classID > 0) )
			{
				void* object = (void*)(hkUlong)( objectID );
				hkClass* klass = (hkClass*)(hkUlong)( classID );

   				hkPointerMap<const void*, const hkClass*> writtenObjects;

				// for now, as a class is pretty useless anyway without recursion
				// we will recurse on classes automatically even if not asked
				bool recurse = (klass == &hkClassClass) || (updateFlags & HK_INSPECT_FLAG_RECURSE);

				/*int bytesWriten = */writeObject( object, *klass, recurse, writtenObjects );
			
				int na = m_autoUpdateList.getSize();
				int a = 0;
				for (; a < na; ++a)
				{
					if (m_autoUpdateList[a].obj == object)
						break;
				}
				bool inAutoList = a != na;
				if ( updateFlags & HK_INSPECT_FLAG_AUTOUPDATE )
				{
					if (!inAutoList)
					{
						ObjectPair& op = m_autoUpdateList.expandOne();
						op.klass = klass;
						op.obj = object;
					}
				}
				else if (inAutoList)
				{
					m_autoUpdateList.removeAt(a);
				}
			}
		}
		break;
	}
}

int hkInspectProcess::addTopLevelObject( void* ptr, const hkClass& klass, hkUlong tag )
{
	int packetSize = 1 + (3*8);
	m_outStream->write32u(packetSize);

	// see if it is really the klass
	const hkClass* derivedClass = &klass;
	if ( m_vdb->getClassReg() && klass.hasVtable() )
	{
		derivedClass = m_vdb->getClassReg()->getClassFromVirtualInstance( ptr );
		if (!derivedClass) // then we will leave at the abstract base we know of.
		{
			derivedClass = &klass;
		}
	}

	m_outStream->write8u(hkVisualDebuggerProtocol::HK_ADD_TOPLEVEL);
	m_outStream->write64u( hkUlong(ptr) ); // ptrID
	m_outStream->write64u( hkUlong(derivedClass) ); // klassID
	m_outStream->write64u( hkUlong(tag) ); // Havok710 : Tag 

	return packetSize;
}

int hkInspectProcess::removeTopLevelObject( void* ptr )
{
	int packetSize = 1 + 8; 
	m_outStream->write32u(packetSize);

	m_outStream->write8u(hkVisualDebuggerProtocol::HK_REMOVE_TOPLEVEL);
	m_outStream->write64u( hkUlong(ptr) ); // ptrID
	// see if it is in our auto update list:
	
	int na = m_autoUpdateList.getSize();
	for (int a=0; a < na; ++a)
	{
		if (m_autoUpdateList[a].obj == ptr)
		{
			m_autoUpdateList.removeAt(a);
			break;
		}
	}

	return packetSize;
}

// write back in PC format
int hkInspectProcess::writeObject( const void* ptr, const hkClass& klass, hkBool recur, hkPointerMap<const void*, const hkClass*>& writtenObjects )
{
	if(!hkString::strCmp(klass.getName(), "hkClass"))
	{
		// Make sure to use hkClassClass here since the provided 'klass' is likely the VDB client's version of the hkClassClass, which will break things if we try and use it for serialization on the VDB server
		return hkObjectSerialize::writeObject(m_outStream, hkVisualDebuggerProtocol::HK_ADD_OBJECT, ptr, hkClassClass, writtenObjects, true, recur, false, m_vdb->getClassReg(), m_cache, hkObjectCopier::FLAG_RESPECT_SERIALIZE_IGNORED);
	}
	else
	{
		return hkObjectSerialize::writeObject(m_outStream, hkVisualDebuggerProtocol::HK_ADD_OBJECT, ptr, klass, writtenObjects, true, recur, false, m_vdb->getClassReg(), m_cache, hkObjectCopier::FLAG_NONE);
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

/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/Process/hkRemoteObjectProcess.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/Serialize/hkObjectSerialize.h>

int hkRemoteObjectProcess::m_tag = 0;

void hkRemoteObjectClientSideListener::sendObject( hkReferencedObject* object )
{
	if( m_sendFunc != HK_NULL )
	{
		m_sendFunc(object);
	}
}

hkRemoteObjectProcess::hkRemoteObjectProcess()
:hkProcess(true)
{
}

hkRemoteObjectProcess::~hkRemoteObjectProcess()
{
}

void hkRemoteObjectProcess::init()
{	
}

void hkRemoteObjectProcess::addListener( hkRemoteObjectServerSideListener* listener )
{
	m_listeners.pushBack(listener);
}

void hkRemoteObjectProcess::removeListener( hkRemoteObjectServerSideListener* listener )
{
	hkInt32 index = m_listeners.indexOf(listener);
	if( index != -1 )	
	{
		m_listeners.removeAt(index);
	}
}

void hkRemoteObjectProcess::sendObject( hkDisplaySerializeOStream* stream, hkReferencedObject* object, SendObjectType objType /*= SEND_OBJECT_PACKFILE*/ )
{
	if( stream != HK_NULL && object != HK_NULL )
	{
		bool writePackfile = (objType == SEND_OBJECT_PACKFILE);
		hkObjectSerialize::writeObject( stream, object, true, writePackfile );
	}
}

static hkUint8 _remoteObject_cmds[] = { hkVisualDebuggerProtocol::HK_LIVE_OBJECT };

void hkRemoteObjectProcess::getConsumableCommands( hkUint8*& commands, int& numCommands )
{
	commands = _remoteObject_cmds;
	numCommands = 1;
}

void hkRemoteObjectProcess::consumeCommand( hkUint8 command )
{
	switch(command)
	{
		case hkVisualDebuggerProtocol::HK_LIVE_OBJECT:
		{
			hkSerializeUtil::ErrorDetails errorDetails;
			hkReferencedObject* object = hkObjectSerialize::readObject(m_inStream, errorDetails);
			const hkClass* klass = object? hkBuiltinTypeRegistry::getInstance().getVtableClassRegistry()->getClassFromVirtualInstance(object) : HK_NULL;
			
			if( (klass != HK_NULL) && (object != HK_NULL) )
			{
				for( int i = 0; i < m_listeners.getSize(); ++i )
				{
					m_listeners[i]->receiveObjectCallback( object, klass );
				}

				object->removeReference();
			}
			else
			{
				if (errorDetails.id != hkSerializeUtil::ErrorDetails::ERRORID_NONE)
				{
					HK_WARN_ALWAYS(0x6345fed, errorDetails.defaultMessage.cString() );
				}
				else
				{
					HK_WARN_ALWAYS(0x6345fed, "Could not load object from network for unknown reason");
				}
			}
		}
		break;
	}
}

hkProcess* HK_CALL hkRemoteObjectProcess::create(const hkArray<hkProcessContext*>& contexts)
{
	return new hkRemoteObjectProcess();
}

void HK_CALL hkRemoteObjectProcess::registerProcess()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
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

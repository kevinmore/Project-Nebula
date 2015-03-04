/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Copier/hkDeepCopier.h>
#include <Common/Base/Reflection/Attributes/hkAttributes.h>
#include <Common/Base/Reflection/Registry/hkTypeInfoRegistry.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Serialize/Serialize/hkRelocationInfo.h>

namespace
{
	struct Copy
	{
		Copy(int f, void* o, const hkClass* c) : copyOffset(f), original(o), klass(c) { }
		Copy() {}
		int copyOffset;
		void* original;
		const hkClass* klass;
	};

	// At the end of each allocated block, we store a list of TypeInfoStore's, followed
	// by the size of the list. As we can easily get the allocation size, we can work
	// backwards to get the list
	struct TypeInfoStore
	{
		const hkTypeInfo* m_typeInfo;
		void* m_pointer;
	};
}

void* hkDeepCopier::deepCopy(const void* dataIn,
							 const hkClass& klassIn,
							 hkDeepCopier::CopyFromOriginal* previousCopies,
							 hkVtableClassRegistry* vtableClassRegistry,
							 hkTypeInfoRegistry* loadedObjectRegistry,
							 hkObjectCopier::ObjectCopierFlags flags )
{
	HK_ASSERT(0x67a10c3b, previousCopies == HK_NULL || previousCopies->hasKey(dataIn) == false );

	hkArray<char> buffer;
	hkOstream out(buffer);

	hkRelocationInfo relocations;
	hkArray<hkVariant> todo;
	hkPointerMap<void*, int> onTodoList;
	{
		hkVariant v;
		v.m_object = const_cast<void*>(dataIn);
		v.m_class =  &klassIn;
		todo.pushBack(v);
		onTodoList.insert(v.m_object,1);
	}

	hkPointerMap<void*, int> offsetFromOriginal; // copies in the current buffer
	hkArray<Copy> copiesMade;

	hkObjectCopier copier( hkStructureLayout::HostLayoutRules, hkStructureLayout::HostLayoutRules, flags );

	for( int todoIndex = 0; todoIndex < todo.getSize(); ++todoIndex )
	{
		void* oldObject = todo[todoIndex].m_object;
		const hkClass* klass = todo[todoIndex].m_class;

		offsetFromOriginal.insert( oldObject, buffer.getSize() );
		int origNumGlobals = relocations.m_global.getSize();

		// look up the base class

		if ( klass->hasVtable() )
		{
			HK_ASSERT( 0x9a2b381c, vtableClassRegistry != HK_NULL );
			const hkClass* actualKlass = vtableClassRegistry->getClassFromVirtualInstance( oldObject );
			HK_ASSERT( 0x40a12b56, actualKlass != HK_NULL );
			klass = actualKlass;
		}

		copiesMade.pushBack( Copy(buffer.getSize(), oldObject, klass) );
		copier.copyObject( oldObject, *klass, out.getStreamWriter(), *klass, relocations );

		for( int i = origNumGlobals; i < relocations.m_global.getSize(); ++i )
		{
			void* obj = relocations.m_global[i].m_toAddress;

			// nulls may be in the relocation list
			if ( obj != HK_NULL && relocations.m_global[i].m_toClass != HK_NULL )
			{
				hkVariant v;
				v.m_object = obj;
				v.m_class =  relocations.m_global[i].m_toClass;

				void* previous = previousCopies ? previousCopies->getWithDefault(v.m_object, HK_NULL) : HK_NULL;
				if( previous )
				{
					relocations.m_global[i].m_toAddress = previous;
				}
				else if( onTodoList.getWithDefault(v.m_object,0) == 0 )
				{
					todo.pushBack( v );
					onTodoList.insert(v.m_object, 1);
				}
			}
		}
	}


	if( buffer.getSize() )
	{
		int extraSize = hkSizeOf(hkInt32);
		if ( loadedObjectRegistry != HK_NULL )
		{
			// Need to reserve extra space to store the object destructors
			extraSize += copiesMade.getSize() * hkSizeOf(TypeInfoStore);
		}
		// hkObjectCopier should pad up the buffer. Test that pointer alignment is ok
		
		HK_ASSERT2(0x5e749ea1, (buffer.getSize() % HK_POINTER_SIZE) == 0, "Unaligned deepcopy buffer");

		char* versioned = hkAllocate<char>( buffer.getSize() + extraSize, HK_MEMORY_CLASS_EXPORT );
		TypeInfoStore* objectList = reinterpret_cast<TypeInfoStore*>(hkAddByteOffset(versioned, buffer.getSize()));
		hkString::memCpy( versioned, buffer.begin(), buffer.getSize() );

		hkArray<hkRelocationInfo::Local>& local = relocations.m_local;
		for( int localIndex = 0; localIndex < local.getSize(); ++localIndex )
		{
			*(void**)(versioned + local[localIndex].m_fromOffset) = versioned + local[localIndex].m_toOffset;
		}

		hkArray<hkRelocationInfo::Global>& global = relocations.m_global;
		for( int globalIndex = 0; globalIndex < global.getSize(); ++globalIndex )
		{
			void* porig = global[globalIndex].m_toAddress;
			void* pnew  = porig;
			int off = offsetFromOriginal.getWithDefault(porig,-1);
			if( off != -1 )
			{
				pnew = versioned + off;
			}
			void* from = versioned + global[globalIndex].m_fromOffset;
			*(void**)(from) = pnew;
		}
		
		if( previousCopies != HK_NULL )
		{
			for( int i = 0; i < copiesMade.getSize(); ++i )
			{
				const Copy& c = copiesMade[i];
				previousCopies->insert( c.original, versioned + c.copyOffset );
			}
		}

		// finish the new copies (set up vtable, etc)

		if ( loadedObjectRegistry != HK_NULL )
		{
			// call finishing constructor
			for( int i = 0; i < copiesMade.getSize(); ++i )
			{
				const Copy& c = copiesMade[i];
				loadedObjectRegistry->finishLoadedObject( versioned + c.copyOffset, c.klass->getName() );
				// Store the typeinfo and object pointer for each one so we can
				// destruct them properly later
				objectList[i].m_typeInfo = loadedObjectRegistry->getTypeInfo( c.klass->getName() );
				objectList[i].m_pointer = versioned + c.copyOffset;
			}

			// call post finish
			for( int i = 0; i < copiesMade.getSize(); ++i )
			{
				const Copy& c = copiesMade[i];
				// Call post finish contructor if it exists
				const hkVariant* attr = c.klass->getAttribute( "hk.PostFinish" );
				if ( attr != HK_NULL )
				{
					HK_ASSERT2( 0x22440204, attr->m_class->equals( &hkPostFinishAttributeClass ), "Object does not have PostFinish attribute" );
					const hkPostFinishAttribute* postFinishAttr = reinterpret_cast<hkPostFinishAttribute*>( attr->m_object );
					postFinishAttr->m_postFinishFunction( versioned + c.copyOffset );
				}
			}

			// Write in the list size
			*reinterpret_cast<hkInt32*>(&objectList[copiesMade.getSize()]) = copiesMade.getSize();
		}
		else
		{
			// Store zero so we know not to try and destruct anything
			*reinterpret_cast<hkInt32*>(objectList) = 0;
		}
		return versioned;
	}
	return HK_NULL;
}

hkResult hkDeepCopier::freeDeepCopy(void* data)
{
	// We know hkAllocate has been used so this is safe to do
	hkMemoryRouter& a = hkMemoryRouter::getInstance();
	hkUlong allocationSize = hkMemoryRouter::getEasyAllocSize(a.heap(), data);

	// The number of objects that must be destructed. This is at the end of the allocation
	hkInt32 listSize = *reinterpret_cast<hkInt32*>(hkAddByteOffset(data, allocationSize - hkSizeOf(hkInt32)));
	if(listSize)
	{
		// Work backwards to find the start of the list, now that we know the size
		int totalExtraSize = hkSizeOf(hkInt32) + (listSize * hkSizeOf(TypeInfoStore));
		TypeInfoStore* objectList = reinterpret_cast<TypeInfoStore*>(hkAddByteOffset(data, allocationSize - totalExtraSize));
		for(int i=0; i<listSize; i++)
		{
			// Find the destructor and call it, if there is one
			if(objectList[i].m_typeInfo)
			{
				objectList[i].m_typeInfo->cleanupLoadedObject(objectList[i].m_pointer);
			}
		}
	}

	// Now that everything has been tidied up, it is safe to free the memory
	hkDeallocate<char>(reinterpret_cast<char*>(data));
	return HK_SUCCESS;
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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

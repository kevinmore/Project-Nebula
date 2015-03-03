/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Platform/WinRT/hkWinRTFileSystem.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Container/String/hkUtf8.h>
#include <Common/Base/System/Io/Platform/Stdio/hkStdioStreamReader.h>
#include <Common/Base/System/Io/Platform/Win32/hkWin32StreamWriter.h>
#include <Common/Base/Thread/Thread/WinRT/hkWinRTThreadUtils.h>

using namespace Windows::Foundation;
using namespace Windows::ApplicationModel;
using namespace Windows::Storage;
using namespace Windows::Foundation::Collections;
namespace WFC = Windows::Foundation::Collections;

using namespace std;

#include <ppl.h>
#include <ppltasks.h>
#include <Strsafe.h> // Metro app  allowed string funcs
using namespace Concurrency;


#define HK_WIDEN(X) hkUtf8::WideFromUtf8(X).cString()
#define HK_NARROW(X) hkUtf8::Utf8FromWide(X).cString()
#define HK_STRCMP(VAR,CONST_STRING) wcscmp(VAR, L ## CONST_STRING)
#include <windows.h>
#include <io.h>

//Keep me to compute the #100ns since the epoch
// 	SYSTEMTIME st = { 1970,1,0,1,   0,0,0,0 };
// 	FILETIME ft;
// 	SystemTimeToFileTime(&st, &ft);
// 	hkInt64 delta = s_combineHiLoDwords( ft.dwHighDateTime, ft.dwLowDateTime );
//  delta == 116444736000000000UI64 * 100ns from win32 epoch to linux epoch

#define HK_TIMESTAMP_NSEC100_TO_UNIX_EPOCH  116444736000000000UI64

static hkUint64 s_convertWindowsFiletimeToUnixTime( DateTime dt)
{
	hkUint64 filetime = dt.UniversalTime; //XX check
	return (filetime - HK_TIMESTAMP_NSEC100_TO_UNIX_EPOCH) * 100;
}

hkRefNew<hkStreamReader> hkWinRTFileSystem::openReader(const char* name, OpenFlags flags)
{
	hkStringBuf sb = name; sb.replace('/', '\\');
	return _handleFlags( hkStdioStreamReader::open(sb), OpenFlags(flags&~OPEN_BUFFERED) );
}

hkRefNew<hkStreamWriter> hkWinRTFileSystem::openWriter(const char* name, OpenFlags flags)
{
	hkStringBuf sb = name; sb.replace('/', '\\');
	int dwCreationDisposition = (flags&OPEN_TRUNCATE) ? CREATE_ALWAYS : OPEN_ALWAYS;
	return _handleFlags( hkWin32StreamWriter::open(sb, dwCreationDisposition ), flags );
}

hkFileSystem::Result hkWinRTFileSystem::remove(const char* path)
{
	hkStringBuf sb = path; sb.replace('/', '\\');
	return DeleteFileW( HK_WIDEN(path) ) ? RESULT_OK : RESULT_ERROR;
}
hkFileSystem::Result hkWinRTFileSystem::mkdir(const char* path)
{
	hkStringBuf sb = path; sb.replace('/', '\\');
	return CreateDirectoryW( HK_WIDEN(sb), HK_NULL ) ? RESULT_OK : RESULT_ERROR;
}


hkFileSystem::Result hkWinRTFileSystem::stat( const char* path, Entry& entryOut )
{
	HK_ASSERT2(0x129e4884, hkString::strChr(path,'*')==0, "Use an iterator for wildcards" );
	
	if ( !Havok::isBlockingAllowed() )
	{
		HK_WARN_ONCE(0x2549f350, "You are calling a WinRT blocking function (hkWinRTFileSystem::stat) from the main UI thread. This is not supported.");
		return RESULT_ERROR;
	}

	try {
	
		hkStringBuf justFileNameA(path); justFileNameA.pathBasename(); 
		hkStringBuf justPathA(path); justPathA.pathDirname(); justPathA.replace('/', '\\');
		Platform::String^ pathW = ref new Platform::String( HK_WIDEN(justPathA) );
		Platform::String^ fileW = ref new Platform::String( HK_WIDEN(justFileNameA) );
		IAsyncOperation<Windows::Storage::StorageFolder^>^ mainFolderOperation = StorageFolder::GetFolderFromPathAsync(pathW);
		task<StorageFolder^> getFolderTask( mainFolderOperation );
		task<Windows::Storage::IStorageItem^ > itemTask = getFolderTask.then( [&](StorageFolder^ resourceFolder)
		{
			return resourceFolder->GetItemAsync(fileW);
		}, Concurrency::task_continuation_context::use_arbitrary() );
					
		task<Windows::Storage::FileProperties::BasicProperties^ > propTask = itemTask.then( [&](IStorageItem^ item) 
		{
			return item->GetBasicPropertiesAsync();
		}, Concurrency::task_continuation_context::use_arbitrary() );
			
		propTask.wait();

		IStorageItem^ item = itemTask.get();
		FileProperties::BasicProperties^ bp = propTask.get();
	
		entryOut.setAll
		(
			this,
			path,
			item->IsOfType( Windows::Storage::StorageItemTypes::Folder ) ? hkFileSystem::Entry::F_ISDIR : hkFileSystem::Entry::F_ISFILE,
			s_convertWindowsFiletimeToUnixTime( bp->DateModified ),
			bp->Size
		);


		return RESULT_OK;
	}
	catch (Platform::Exception^ e)
	{
		// dir not exist etc
	}

	return RESULT_ERROR;
}

namespace
{
	struct WinRTImpl : public hkFileSystem::Iterator::Impl
	{
		WinRTImpl(hkFileSystem* fs, const char* top, const char* wildcard)
			: m_fs(fs)
			, m_top(top)
			, m_wildcard(wildcard)
			, m_currentIndex(0)
			, m_items(nullptr)
		{
			HK_ASSERT2(0x1e0bb0cd, hkString::strChr(m_top,'*') == HK_NULL, "Path part cannot contain a *");
			HK_ASSERT2(0x47f9de01, wildcard==HK_NULL || hkString::strChr(wildcard,'*'), "Wildcard must be null or contain a *" );

		}

		virtual bool advance(hkFileSystem::Entry& e)
		{
			if( m_items == nullptr ) 
			{
				if ( !Havok::isBlockingAllowed() )
				{
					HK_WARN_ONCE(0x2549f350, "You are calling a WinRT blocking function (hkFileSystem::Iterator) from the main UI thread. This is not supported.");
					return false;
				}

				try {
	
					hkStringBuf pathA(m_top); pathA.replace('/', '\\'); pathA.replace("\\.\\", "\\");
					Platform::String^ path = ref new Platform::String( HK_WIDEN(pathA) );
					IAsyncOperation<Windows::Storage::StorageFolder^>^ mainFolderOperation = StorageFolder::GetFolderFromPathAsync(path);
		
					task<StorageFolder^> getFolderTask( mainFolderOperation );
					task< IVectorView< Windows::Storage::IStorageItem^>^ > itemsTask = getFolderTask.then( [&](StorageFolder^ resourceFolder)
					{
						return resourceFolder->GetItemsAsync();
					}, Concurrency::task_continuation_context::use_arbitrary() );
			
					itemsTask.wait();
					m_items = itemsTask.get();
					m_currentIndex = 0;
				}
				catch (...)
				{
					// dir not exist etc
					m_items = nullptr;
					return false;
				}
			}
				
			while ( m_items != nullptr && (m_currentIndex < m_items->Size) )
			{
				Windows::Storage::IStorageItem^ item = m_items->GetAt(m_currentIndex);
				hkStringPtr itemName = HK_NARROW( item->Name->Data() );
				if( hkFileSystem::Iterator::nameAcceptable( itemName.cString(), m_wildcard.cString() ) )
				{
					//XX Should have a flag perhaps on advance to see if this info really needed:
					try {
						task<Windows::Storage::FileProperties::BasicProperties^ > propTask( item->GetBasicPropertiesAsync() );
						propTask.wait();
						FileProperties::BasicProperties^ bp = propTask.get();
						e.setAll
						(
							m_fs,
							hkStringBuf(m_top).pathAppend( itemName ),
							item->IsOfType( Windows::Storage::StorageItemTypes::Folder ) ? hkFileSystem::Entry::F_ISDIR : hkFileSystem::Entry::F_ISFILE,
							s_convertWindowsFiletimeToUnixTime( bp->DateModified ),
							bp->Size
						);
					
						++m_currentIndex;
						return true;
					}
					catch (...)
					{
						// not allowed or file since gone, will try next?
					}
				}
		
				// skipped, or failed, so try next
				++m_currentIndex;
			}
			return false;
		}

		~WinRTImpl()
		{
			m_items = nullptr;
		}

		hkFileSystem* m_fs;
		hkStringPtr m_top;
		hkStringPtr m_wildcard;

		unsigned int m_currentIndex;
		IVectorView< Windows::Storage::IStorageItem^ >^ m_items; 
	};
}

hkRefNew<hkFileSystem::Iterator::Impl> hkWinRTFileSystem::createIterator( const char* top, const char* wildcard )
{
	return new WinRTImpl(this, top, wildcard);
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

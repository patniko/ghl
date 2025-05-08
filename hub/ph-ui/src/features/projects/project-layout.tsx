import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { Search } from '@/components/search'
import { ThemeSwitch } from '@/components/theme-switch'
import { Outlet } from '@tanstack/react-router'

export default function ProjectLayout() {
  // We don't need the params in the layout, but keeping this for reference
  // const { orgSlug, projectName } = useParams({ 
  //   from: '/_authenticated/$orgSlug/$projectName' 
  // })
  
  return (
    <>
      <Header fixed>
        <Search />
        <div className="ml-auto flex items-center space-x-4">
          <ThemeSwitch />
          <ProfileDropdown />
        </div>
      </Header>

      <Main>
        {/* Child routes will be rendered here */}
        <Outlet />
      </Main>
    </>
  )
}
